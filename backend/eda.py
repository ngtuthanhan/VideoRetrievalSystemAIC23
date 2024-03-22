import numpy as np
import json
import glob
import os
feature = np.load("./retriever/feature.npy")
index = np.load("./retriever/index.npy")
with open("./data/keyframe.json", "r") as outfile:
    keyframes = json.load(outfile)
print(np.linalg.norm(feature, axis=-1, keepdims=True))
root_data = "/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023"
all_keyframes = glob.glob(os.path.join(root_data,'**','keyframes', '**', '*.jpg'))
all_videos = glob.glob(os.path.join(root_data,'**','video', '*.mp4'))
all_map_keyframes = glob.glob(os.path.join(root_data,'**','map-keyframes', '*.csv'))
all_data_features = glob.glob(os.path.join(root_data,'**','clip-features', '*.npy'))
all_data_features += glob.glob(os.path.join(root_data,'**','clip-features-32', '*.npy'))
print(len(all_keyframes))
print(len(index), len(feature))
print(len(keyframes))
print(len(all_videos), len(all_map_keyframes), len(all_data_features))