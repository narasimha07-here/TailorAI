import pandas as pd
from fastai.vision.all import *
import numpy as np

df1=pd.read_csv(r"Dataset\hwg_metadata.csv")
df2=pd.read_csv(r"Dataset\subject_to_photo_map.csv")
df3=pd.merge(df2,df1,on="subject_id")
df4=pd.read_csv(r"Dataset\measurements.csv")
df=pd.merge(df3,df4,on="subject_id")

path=r"Dataset"
def get_front(row):
    frontal=f"{path}/mask/{row['photo_id']}.png"
    return PILImage.create(frontal)

def get_side(row):
    lateral=f"{path}/mask_left/{row['photo_id']}.png"
    return PILImage.create(lateral)

def get_Y(row):
    # List of columns to extract
    columns = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height", "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist"]
    dim = row[columns].values.astype(float)
    return dim


def get_dataloaders():

    dls=DataBlock(
        blocks=(ImageBlock,ImageBlock,RegressionBlock),
        get_x=[get_front,get_side],
        get_y=get_Y,
        splitter=RandomSplitter(0.2,seed=42),
        item_tfms=Resize(320),
        batch_tfms=aug_transforms())

    return dls.dataloaders(df,bs=32)