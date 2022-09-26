import os,sys,torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from models.sasv import SASV
from dataset.sasv_dataset import get_dataset, get_eval_dataset
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin,overrides)
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    ## fix
    for i in hparams["embedding_model"].parameters():
        i.requires_grad=False
    

    datasets = get_dataset(hparams)
    evalsets = get_eval_dataset(hparams)
    sasv_model = SASV(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    sasv_model.fit(
        epoch_counter=sasv_model.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=evalsets["dev"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["validloader_options"],
    )
