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
with open("./data/keyframe_full.json", "r") as file:
    data = json.load(file)

for keyframe in tqdm(data[-4:]):
    print(keyframe)
   
    # keyframe_json.append({
    #     "OCR": ocr_sentence,
    #     "ASR": asr_sentence,
    #     "path": path,
    #     "youtube_id": youtube_id,
    #     "pts_time": pts_time,
    #     "video": video,
    #     "keyframe":video + "_" + str(keyframe_idx),
    #     "keyframe_idx": int(keyframe_idx),
    #     "keyframe_position": int(keyframe_position)
    # })


# with open("./data/keyframe_full.json", "w") as outfile:
#     json.dump(keyframe_json, outfile)

# # with open('./data/clip-feature_full.npy','wb') as f:
# #     np.save(f, clip_features)

# # with open('./data/hist-feature_full.npy','wb') as f:
# #     np.save(f, hist_features)


# with open("./data/keyframe_full.json", "r") as file:
#     data = json.load(file)

# # Convert the data to a list of bulk actions with newline separators
# batch_size=6000

# for i in tqdm(range(0, len(data), batch_size)):
#     bulk_actions = []
#     for j in range(i, i+batch_size):
#         if j >= len(data):
#             break
#         item = data[j]
#         # Each action consists of two lines: one for the action metadata and one for the document data
#         bulk_actions.append(json.dumps({"index": {"_index": "test", "_type": "_doc", "_id": str(j)}}))
#         bulk_actions.append(json.dumps(item))
#     # Join the bulk actions with newline characters
#     bulk_request = "\n".join(bulk_actions) + "\n"

#     # Save the bulk request to a new file
#     with open(f"./data/keyframe_split/bulk_keyframe_batch_{i}.json", "w") as file:
#         file.write(bulk_request)
