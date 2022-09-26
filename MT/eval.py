import pandas as pd
import torch, json, os, sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from models.sasv import SASV
from dataset.sasv_dataset import get_eval_dataset
import logging
from collections import defaultdict
from pathlib import Path
from utils.metrics import get_sasv_scores, get_verification_scores
from utils.visualize import visualize, reduce_dimension
from utils.save_embeddings import save_embedding
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

DATA_DIR = 'spk_meta'

OUTPUT_DIR = 'predictions'
Path(OUTPUT_DIR).mkdir(exist_ok=True)

if __name__ == "__main__":

    VISUALIZE = False
    SCORE = True

    logger = logging.getLogger(__name__)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams_file = os.path.join(hparams['output_folder'], 'hyperparams.yaml')
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams['validloader_options']['batch_size'] = 64

    datasets = get_eval_dataset(hparams)

    encoder = SASV(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    save_file_dict = {
        "test_asv_emb_file"  :os.path.join(hparams['output_folder'],'test_asv_emb.pt'),
        "test_sasv_scores_file":os.path.join(hparams['output_folder'],'test_sasv_scores.pt'),
        "test_cm_emb_file"   :os.path.join(hparams['output_folder'],'test_cm_emb.pt')
    }  
    
    if not os.path.exists(save_file_dict["test_asv_emb_file"]):
        save_embedding(datasets, encoder, hparams, save_file_dict)

    test_sasv_scores_dict = torch.load(save_file_dict["test_sasv_scores_file"])
    test_asv_emb_dict   = torch.load(save_file_dict["test_asv_emb_file"])
    test_cm_emb_dict    = torch.load(save_file_dict["test_cm_emb_file"])  
        
    with open('protocols/asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt','r') as f:
        test_data = f.readlines()
    # with open(os.path.join(DATA_DIR, ENROLL_FILE)) as f:
    #     enroll_data = json.load(f)

    if SCORE:
        # get_verification_scores(test_data, enroll_data, test_asv_emb_dict, enrol_asv_emb_dict, test_cm_scores_dict)
        get_sasv_scores(test_data, test_sasv_scores_dict)

    if VISUALIZE:

        if os.path.exists(os.path.join(hparams['output_folder'], 'all_2d.csv')):
            df = pd.read_csv(os.path.join(hparams['output_folder'], 'all_2d.csv'))
            visualize(df=df, show="cm")
        else:
            for utt_id in test_asv_emb_dict:
                test_asv_emb_dict[utt_id] = torch.squeeze(test_asv_emb_dict[utt_id],0)
                test_asv_emb_dict[utt_id] = torch.squeeze(test_asv_emb_dict[utt_id],0)
                test_asv_emb_dict[utt_id] = test_asv_emb_dict[utt_id].cpu().numpy()

            # process all data
            gt_data = defaultdict(dict)
            for utt in test_data:
                speaker_id, utt_id, cm_id, target = utt.split()
                gt_data[utt_id]['speaker_id'] = speaker_id
                gt_data[utt_id]['cm_id'] = cm_id

            bonafide_dict = {}
            for utt_id in test_asv_emb_dict:
                if gt_data[utt_id]['cm_id'] == 'bonafide':
                    bonafide_dict[utt_id] = test_asv_emb_dict[utt_id]

            reduce_dimension(test_asv_emb_dict, gt_data, 'all_2d.csv', hparams)
            reduce_dimension(bonafide_dict, gt_data, 'spk_2d.csv', hparams)

            df = pd.read_csv('results/sasv/1678/all_2d.csv')
            spk = pd.read_csv('results/sasv/1678/spk_2d.csv')
            visualize(df, spk, show="cm")
            
    del test_asv_emb_dict, test_cm_emb_dict, test_sasv_scores_dict