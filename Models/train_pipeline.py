import torch.nn as nn
from torch.nn.parameter import Parameter
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.model import SequenceModel
from Models.Data_loaders import get_dataloaders
from fastai.vision.all import *

loss= nn.SmoothL1Loss()
dls = get_dataloaders()
measurements = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height", 
               "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", 
               "waist", "wrist"]

modele = SequenceModel(3,0.4)

def get_train():
    learn1=Learner(dls,model=modele,loss_func=loss,metrics=mae)
    learn1.fit_one_cycle(15)


if __name__ == "__main__":
    get_train()
