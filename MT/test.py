import argparse
import os,yaml
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.strategies.ddp import DDPStrategy

from utils.utils import *
from models.rawnet3 import MainModel

warnings.filterwarnings("ignore", category=FutureWarning)

model_name = "/home/alex/Speech/SASV/SASVC2022_Baseline/RawNet3/model.pt"
with open("yaml/RawNet3_AAM.yaml", "r") as f:
    yml_config = yaml.load(f, Loader=yaml.FullLoader)

os.environ["CUDA_VISABLE_DEVICE"] = "1"
model = MainModel(**yml_config)
load_parameters(model.state_dict(), model_name)
model.eval()
print(model)
