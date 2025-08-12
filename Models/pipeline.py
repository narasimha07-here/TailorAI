from fastai.vision.all import *
import torch
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

measurements = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height", 
               "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", 
               "waist", "wrist"]

class Measurements():
    def __init__(self):
        self.learner = load_learner("Models/Trained_model/ragnet.pkl")
        self.preprocessing()

    def preprocessing(self):
        self.preprocess = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

    def predict(self, front_image, side_image):
        if isinstance(front_image, str):
            front_image = Image.open(front_image).convert("RGB")

        if isinstance(side_image, str):
            side_image = Image.open(side_image).convert("RGB")

        frontal = self.preprocess(front_image).unsqueeze(0)
        lateral = self.preprocess(side_image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.learner.model(frontal, lateral)

            pred = predictions.squeeze(0).tolist()

        return {measurements[i] : pred[i] for i in range(len(measurements))}
