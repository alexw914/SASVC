import speechbrain as sb
import os
import torch
from dataset.speech_process import load_wav

SPEAKER_ID_LIST = 'speaker_id_list.json'

def get_asv_eval_dataset():
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        sig = load_wav(file_path)
        return sig

    data_dir = 'spk_meta'
    enrol_file = 'eval_enroll.json'
    trl_file = 'eval_trl.json'

    datasets = {}

    datasets['enrol'] = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= os.path.join(data_dir, enrol_file),
        dynamic_items=[audio_pipeline,
                       ],
        output_keys=["id", "sig", "speaker_id"],
    )

    datasets['trl'] = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= os.path.join(data_dir, trl_file),
        dynamic_items=[audio_pipeline,
                        ],
        output_keys=["id", "sig", "speaker_id"],
    )
    return datasets
