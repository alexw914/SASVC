import torch,os
import speechbrain as sb
import torch.nn as nn
from torch.utils.data import DataLoader
from speechbrain import Stage
from speechbrain.nnet.linear import Linear
from tqdm.contrib import tqdm
import torch.nn.functional as F
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from pytorch_model_summary import summary
from models.BinaryMetricStats import BinaryMetricStats

class SASV(sb.Brain):

    def compute_forward(self, batch, stage):

        batch = batch.to(self.device)
        fbanks = self.prepare_features(batch.sig, stage)
        feature, asv_emb = self.modules.embedding_model(fbanks)

        enrol_fbanks = self.prepare_features(batch.enrol_sig, stage)
        _, enrol_emb = self.modules.embedding_model(enrol_fbanks)

        cm_emb = self.modules.cm_asp_layer(feature)
        sasv_output = self.modules.sasv_model(enrol_emb, asv_emb, cm_emb)

        return (asv_emb, cm_emb, sasv_output)

    def prepare_features(self, wavs, stage):
        wavs, lens = wavs
        if stage == sb.Stage.TRAIN:
            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):
                # Apply augment
                wavs_aug = augment(wavs, lens)
                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig
                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs
            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)
            
        feat_fbank = self.modules.compute_features(wavs)
        feat_fbank = self.modules.mean_var_norm(feat_fbank, lens)

        return feat_fbank

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        _, lens = batch.sig
        # Concatenate labels (due to data augmentation)
        asv_output, cm_output, sasv_output = predictions
        sasv_encoded, _ = batch.sasv_encoded
        
        if stage == sb.Stage.TRAIN:
            bonafide_encoded, _ = batch.bonafide_encoded
            speaker_encoded, _ = batch.speaker_encoded

            speaker_encoded = torch.cat([speaker_encoded]*self.n_augment, dim=0)
            bonafide_encoded = torch.cat([bonafide_encoded]*self.n_augment, dim = 0)
            sasv_encoded = torch.cat([sasv_encoded]*self.n_augment, dim = 0)
            lens = torch.cat([lens, lens])

            asv_loss, prec1 = self.hparams.asv_loss_metric(torch.squeeze(asv_output,1), torch.squeeze(speaker_encoded,1))
            cm_loss, cm_score = self.modules.cm_loss_metric(torch.squeeze(cm_output,1), torch.squeeze(bonafide_encoded,1),is_train=True)
        sasv_loss = self.modules.sasv_loss_metric(torch.squeeze(sasv_output,1), torch.squeeze(sasv_encoded,1))

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            sasv_output = torch.softmax(sasv_output, dim=-1)
            sasv_score = sasv_output[:, 1].unsqueeze(1)
            self.error_metrics.append(batch.id, sasv_score, sasv_encoded)

            return sasv_loss
        loss =  asv_loss + sasv_loss + cm_loss
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = BinaryMetricStats(
                positive_label=1,
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            self.error_metrics.summarize()
            stats = {
                "loss": stage_loss,
                "eer": self.error_metrics.summary["EER"],
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_scheduler(current_epoch = epoch)

            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats,
                                                 num_to_keep=5,
                                                 keep_recent=False,
                                                 min_keys=["eer"])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def evaluate_batch(self, batch, stage):
        """
        Overwrite evaluate_batch.
        Keep same for stage in (TRAIN, VALID)
        Output probability in TEST stage (from classify_batch)
        """

        if stage != sb.Stage.TEST:
            # Same as before
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
            return loss.detach().cpu()
        else:
            asv_emb, cm_emb, sasv_output = self.compute_forward(batch, stage=stage)
            # cm_loss,cm_score = self.modules.cm_loss_metric(asv_emb, None, is_train=False)
            sasv_output = torch.softmax(sasv_output, dim=-1)
            sasv_output = sasv_output[:, 1].unsqueeze(1)
            return asv_emb, cm_emb, sasv_output

    def evaluate(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
    ):
        """
        Overwrite evaluate() function so that it can output score file
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()

        """
        added here
        """
        sasv_score_dict = {}
        cm_emb_dict = {}
        asv_emb_dict = {}

        with torch.no_grad():
            for batch in tqdm(
                    test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                """
                Rewrite here
                """
                asv_emb, cm_emb, sasv_output = self.evaluate_batch(batch, stage=Stage.TEST)
                sasv_scores = [sasv_output[i].item()
                             for i in range(sasv_output.shape[0])]
                for i, seg_id in enumerate(batch.id):
                    asv_emb_dict[seg_id] = asv_emb[i].detach().clone()
                    cm_emb_dict[seg_id] = cm_emb[i].detach().clone()
                    sasv_score_dict[seg_id] = sasv_scores[i]

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

        self.step = 0
        return asv_emb_dict, cm_emb_dict, sasv_score_dict

class ASV_Decoder(torch.nn.Module):

    def __init__(
            self,
            input_size,
            device="cpu",
            lin_blocks=0,
            lin_neurons=192,
            out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)

class SEModule(nn.Module):

    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 192),
            nn.LeakyReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )
        self.se = SEModule(channels=64,bottleneck=32)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc2 = nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,64),
            nn.LeakyReLU(inplace=True),
        )
        self.fc_out = nn.Linear(64,2)

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm, labels=None):

        # asv_enr = torch.unsqueeze(embd_asv_enr, -1) # shape: (bs, 192)
        # asv_tst = torch.unsqueeze(embd_asv_tst, -1) # shape: (bs, 192)
        # cm_tst = torch.unsqueeze(embd_cm, -1) # shape: (bs, 160)
        cm_tst = self.fc(embd_cm)
        x = torch.cat((embd_asv_enr,embd_asv_tst,cm_tst),1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = self.fc_out(x)  # (bs, 2)    
        return x
