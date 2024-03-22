import os
import json
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

if not os.path.exists('./data'):
    os.makedirs('./data')

root_data = "/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023"

# all_keyframes = glob.glob(os.path.join(root_data,'**','keyframes', '**', '*.jpg'),recursive=True)
# all_videos = glob.glob(os.path.join(root_data,'**','video', '*.mp4'),recursive=True)
all_map_keyframes = glob.glob(os.path.join(root_data,'data-batch-1','map-keyframes', '*.csv')) +  glob.glob(os.path.join(root_data,'data-batch-2','map-keyframes', '*.csv')) + glob.glob(os.path.join(root_data,'data-batch-3','map-keyframes', '*.csv'))
keyframe_json = []

clip_features = np.empty([0,512],dtype = float)
hist_features = np.empty([0,3],dtype = float)

for map_keyframes in tqdm(all_map_keyframes):
    video = map_keyframes.split('/')[-1].replace('.csv', '')
    
    df = pd.read_csv(map_keyframes)
    keyframe_name =  map_keyframes.split('/')[-3]
    all_keyframes_in_this_video = glob.glob(os.path.join(root_data,keyframe_name,'keyframes', video, '*.jpg'),recursive=True)
    all_keyframes_in_this_video.sort()

    batch_no = map_keyframes[len('/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-')]
    for path in all_keyframes_in_this_video:
        keyframe_position_str = path.replace('.jpg','').split('/')[-1]
        keyframe_position= int(keyframe_position_str)
        keyframe_idx = list(df[df['n'] == keyframe_position]['frame_idx'])[0]
        pts_time = list(df[df['n'] == keyframe_position]['pts_time'])[0]
        
        if path == '/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-1/keyframes/L01_V001/0003.jpg':
            keyframe_idx = 271
        keyframe = video + '_' + str(keyframe_idx)

        hist_feature = np.load(f'./data/color_vector_full/{keyframe}.npy').reshape((1, -1))
        hist_feature /= np.linalg.norm(hist_feature, axis=-1, keepdims=True)
        hist_features = np.concatenate((hist_features,hist_feature),axis = 0)
        try:
            hist_feature = np.load(f'./data/color_vector_full/{keyframe}.npy').reshape((1, -1))
            # Optionally, normalize the hist_feature
            # hist_feature /= np.linalg.norm(hist_feature, axis=-1, keepdims=True)
        except Exception as e:
            print(f"Error loading or reshaping hist_feature for keyframe {keyframe}: {e}")
            hist_feature = np.zeros((1, 512))  # Handle the error by assigning a default value


# with open("./data/keyframe_full.json", "w") as outfile:
#     json.dump(keyframe_json, outfile)

# with open('./data/clip-feature_full.npy','wb') as f:
#     np.save(f, clip_features)

with open('./data/hist-feature_full.npy','wb') as f:
    np.save(f, hist_features)

