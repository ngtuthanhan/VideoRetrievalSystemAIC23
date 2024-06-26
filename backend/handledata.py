import os
import json
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

if not os.path.exists('./data'):
    os.makedirs('./data')

if not os.path.exists('./data/keyframe_split'):
    os.makedirs('./data/keyframe_split') 
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
    if 'data-batch-1' == keyframe_name or 'data-batch-3' == keyframe_name :
        video_feature_file = map_keyframes.replace('map-keyframes', 'clip-features-vit-b32').replace('.csv','.npy')
    elif 'data-batch-2' == keyframe_name:
        video_feature_file = map_keyframes.replace('map-keyframes', 'clip-features-32').replace('.csv','.npy')
    if '/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-2/clip-features-32/L20_V010.npy' == video_feature_file:
        continue
    clip_feature = np.load(video_feature_file)

    clip_features = np.concatenate((clip_features,clip_feature),axis = 0)

    metadata_path = map_keyframes.replace('map-keyframes', 'metadata').replace('.csv', '.json')
    with open(metadata_path, "r") as f:
        metadata_reader = json.load(f)
    youtube_link = metadata_reader['watch_url']
    youtube_id = youtube_link[youtube_link.find('?v=') + 3:]

    batch_no = map_keyframes[len('/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-')]
    for path in all_keyframes_in_this_video:
        
        keyframe_position_str = path.replace('.jpg','').split('/')[-1]
        keyframe_position= int(keyframe_position_str)
        
        keyframe_idx = list(df[df['n'] == keyframe_position]['frame_idx'])[0]
        pts_time = list(df[df['n'] == keyframe_position]['pts_time'])[0]
        
        if path == '/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-1/keyframes/L01_V001/0003.jpg':
            keyframe_idx = 271
        
        keyframe = video + '_' + str(keyframe_idx)
        asr_path = '/mlcv1/WorkingSpace/Personal/tunglx/AIC23/VideoRetrieval/backend/data/asr_full/' +  keyframe + '.txt'
        ocr_path = '/mlcv1/WorkingSpace/Personal/thuongpt/OCR/Batch_' + batch_no + '/' + video + '/' + keyframe_position_str + '.txt' 
        
        # try:
        #     with open(asr_path, 'r') as f:
        #         asr_sentence =f.read()
        # except Exception as e:
        #     print(f"Error reading ASR file {asr_path}: {e}")
        #     asr_sentence = ""  # Handle the error by appending an empty string
        asr_sentence = ""
        # try:
        #     with open(ocr_path, 'r') as f:
        #         ocr_sentence = f.read()
        # except Exception as e:
        #     print(f"Error reading OCR file {ocr_path}: {e}")
        #     ocr_sentence = "" # Handle the error by appending an empty string
        ocr_sentence = ""
        # hist_feature = np.load(f'./data/color_vector_full/{keyframe}.npy').reshape((1, -1))
        # hist_feature /= np.linalg.norm(hist_feature, axis=-1, keepdims=True)
        # hist_features = np.concatenate((hist_features,hist_feature),axis = 0)
        # try:
        #     hist_feature = np.load(f'./data/color_vector_full/{keyframe}.npy').reshape((1, -1))
        #     # Optionally, normalize the hist_feature
        #     # hist_feature /= np.linalg.norm(hist_feature, axis=-1, keepdims=True)
        # except Exception as e:
        #     print(f"Error loading or reshaping hist_feature for keyframe {keyframe}: {e}")
        #     hist_feature = np.zeros((1, 512))  # Handle the error by assigning a default value

        if path == '/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-1/keyframes/L01_V001/0003.jpg':
            keyframe_idx = 271
        keyframe_json.append({
            "OCR": ocr_sentence,
            "ASR": asr_sentence,
            "path": path,
            "youtube_id": youtube_id,
            "pts_time": pts_time,
            "video": video,
            "keyframe":video + "_" + str(keyframe_idx),
            "keyframe_idx": int(keyframe_idx),
            "keyframe_position": int(keyframe_position)
        })


with open("./data/keyframe_full.json", "w") as outfile:
    json.dump(keyframe_json, outfile)

with open('./data/clip-feature_full.npy','wb') as f:
    np.save(f, clip_features)

# with open('./data/hist-feature_full.npy','wb') as f:
#     np.save(f, hist_features)


with open("./data/keyframe_full.json", "r") as file:
    data = json.load(file)

# Convert the data to a list of bulk actions with newline separators
batch_size=6000

for i in tqdm(range(0, len(data), batch_size)):
    bulk_actions = []
    for j in range(i, i+batch_size):
        if j >= len(data):
            break
        item = data[j]
        # Each action consists of two lines: one for the action metadata and one for the document data
        bulk_actions.append(json.dumps({"index": {"_index": "test", "_type": "_doc", "_id": str(j)}}))
        bulk_actions.append(json.dumps(item))
    # Join the bulk actions with newline characters
    bulk_request = "\n".join(bulk_actions) + "\n"

    # Save the bulk request to a new file
    with open(f"./data/keyframe_split/bulk_keyframe_batch_{i}.json", "w") as file:
        file.write(bulk_request)
