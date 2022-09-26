import speechbrain as sb
import numpy as np
import os, torch
import json,librosa
import pickle as pk
import random
from dataset.speech_process import load_wav

LABEL_DIR = 'spk_meta'
SPEAKER_ID_LIST = 'speaker_id_list.json'


def get_dataset(hparams):
    """
    Code here is basically same with code in SpoofSpeechDataset.py
    However, audio will not be load directly.
    A random compression will be made before load by torchaudio
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder_cm = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_asv = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("base_dir","id")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(base_dir, id):
        sig = load_wav(os.path.join(base_dir, id+".flac"))
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("speaker_id")
    @sb.utils.data_pipeline.provides("speaker_id", "speaker_encoded")
    def speaker_label_pipeline(speaker_id):
        yield speaker_id
        speaker_encoded = label_encoder_asv.encode_label_torch(speaker_id, True)
        yield speaker_encoded

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("bonafide")
    @sb.utils.data_pipeline.provides("bonafide", "bonafide_encoded")
    def bonafide_label_pipeline(bonafide):
        yield bonafide
        bonafide_encoded = label_encoder_cm.encode_label_torch(bonafide, True)
        yield bonafide_encoded


    @sb.utils.data_pipeline.takes("speaker_id","bonafide", "spk_meta_path","base_dir")
    @sb.utils.data_pipeline.provides("enrol_sig", "sasv_encoded")
    def sasv_pipeline(speaker_id, bonafide, spk_meta_path, base_dir):

        with open(spk_meta_path, "rb") as f:
            spk_meta=pk.load(f)

        if bonafide == "bonafide":
            ans_type = random.randint(0, 1)
            if ans_type == 1:  # target
                enrol_id = random.choice(spk_meta[speaker_id]["bonafide"])

            if ans_type == 0:  # zero-effort nontarget
                del spk_meta[speaker_id]
                spk = random.sample(spk_meta.keys(), 1)
                enrol_id = random.choice(spk_meta[spk[0]]["bonafide"])

        if bonafide == "spoof":  # spoof nontraget
            spk = random.sample(spk_meta.keys(),1)
            enrol_id = random.choice(spk_meta[spk[0]]["bonafide"])
            ans_type = 0

        ernrol_sig = load_wav(os.path.join(base_dir, enrol_id+".flac"))
        yield ernrol_sig
        sasv_encoded = torch.LongTensor([ans_type])
        yield sasv_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}

    for dataset in ["train", "dev"]:
        # print(hparams[f"{dataset}_annotation"])
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[
                           audio_pipeline,
                           speaker_label_pipeline,
                           bonafide_label_pipeline,
                           sasv_pipeline,
                           ],
            output_keys=["id", "sig",
                         "bonafide", "bonafide_encoded",
                         "speaker_id", "speaker_encoded",
                         "enrol_sig", "sasv_encoded"
                         ],
        )

    def load_create_label_encoder(label_enc_file,
                                  label_encoder,
                                  label_list_file = None,
                                  ):
        label_list = None
        if label_list_file == None:
            label_list = [("spoof", "bonafide")]
        else:
            with open(os.path.join(LABEL_DIR, label_list_file), 'r') as f:
                label_list = [tuple(json.load(f))]
        lab_enc_file = os.path.join(hparams["save_folder"], label_enc_file)
        label_encoder.load_or_create(
            path=lab_enc_file,
            sequence_input=False,
            from_iterables=label_list,
        )

    load_create_label_encoder(label_enc_file="label_encoder_speaker.txt",
                              label_encoder=label_encoder_asv,
                              label_list_file=SPEAKER_ID_LIST
                              )
    label_encoder_asv.add_unk()

    load_create_label_encoder(label_enc_file="label_encoder_cm.txt",
                              label_encoder=label_encoder_cm,
                              )


    return datasets

def get_eval_dataset(hparams):

    @sb.utils.data_pipeline.takes("base_dir","test_id")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(base_dir, id):
        sig = load_wav(os.path.join(base_dir, id+".flac"))
        return sig


    @sb.utils.data_pipeline.takes("test_speaker_id","spk_meta_path","base_dir", "sasv_label")
    @sb.utils.data_pipeline.provides("enrol_sig","sasv_encoded")
    def sasv_pipeline(test_speaker_id, spk_meta_path, base_dir, sasv_label):

        with open(spk_meta_path, "rb") as f:
            spk_meta=pk.load(f)

        enrol_id = random.choice(spk_meta[test_speaker_id]["bonafide"])

        ernrol_sig = load_wav(os.path.join(base_dir, enrol_id+".flac"))
        yield ernrol_sig
        ans_type = 1 if sasv_label == "target" else 0
        sasv_encoded = torch.LongTensor([ans_type])
        yield sasv_encoded        

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}

    for dataset in ["dev", "eval"]:
        # print(hparams[f"{dataset}_annotation"])
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[
                           audio_pipeline,
                           sasv_pipeline,
                           ],
            output_keys=["id", "sig",
                        "enrol_sig",
                        "sasv_encoded"
                         ],
        )

    return datasets