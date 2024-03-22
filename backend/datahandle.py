# from retriever import extractor
import os
import json
import glob
import pandas as pd
from tqdm import tqdm

# feature, index = extractor()

if not os.path.exists('./data'):
    os.makedirs('./data')

root_data = "/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023"
all_keyframes = glob.glob(os.path.join(root_data,'**','keyframes', '**', '*.jpg'))
all_videos = glob.glob(os.path.join(root_data,'**','video', '*.mp4'))
all_map_keyframes = glob.glob(os.path.join(root_data,'**','map-keyframes', '*.csv'))
keyframe_json = []
i = 0
for map_keyframes in tqdm(all_map_keyframes):
    if i == 10 : break
    else :
        i+=1
        video = map_keyframes.split('/')[-1].replace('.csv', '')

        df = pd.read_csv(map_keyframes)

        all_keyframes_in_this_video = glob.glob(os.path.join(root_data,'**','keyframes', video, '*.jpg'))
        for path in all_keyframes_in_this_video:
            keyframe_position= int(path.replace('.jpg','').split('/')[-1])
            keyframe_idx = list(df[df['n'] == keyframe_position]['frame_idx'])[0]

            if path == '/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-1/keyframes/L01_V001/0003.jpg':
                keyframe_idx = 271
            keyframe_json.append({
                "path": path,
                "video": video,
                "keyframe":video + "_" + str(keyframe_idx),
                "keyframe_idx": int(keyframe_idx),
                "keyframe_position": int(keyframe_position)
            })
    
with open("./data/test_keyframe.json", "w") as outfile:
    json.dump(keyframe_json, outfile)