import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

from .model import SequenceModel

measurements = [
    "ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height",
    "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh",
    "waist", "wrist"
]

class Measurements():
    def __init__(self):

        self.model = SequenceModel(in_features=3, dropout=0.3)
        self.model.load_state_dict(
            torch.load("Models/Trained_model/ragnet.pth", map_location="cpu")
        )
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])

    def predict(self, front_image, side_image):

        if isinstance(front_image, str):
            front_image = Image.open(front_image).convert("RGB")
        else:
            front_image = front_image.convert("RGB")

        if isinstance(side_image, str):
            side_image = Image.open(side_image).convert("RGB")
        else:
            side_image = side_image.convert("RGB")

        frontal = self.preprocess(front_image).unsqueeze(0)
        lateral = self.preprocess(side_image).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(frontal, lateral)
            pred = pred.squeeze(0).tolist()

        return {measurements[i]: pred[i] for i in range(len(measurements))}
