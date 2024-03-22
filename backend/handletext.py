import os
import json
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_keyframe_asr = []
all_keyframe_w_video = []

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('keepitreal/vietnamese-sbert')
model = AutoModel.from_pretrained('keepitreal/vietnamese-sbert')
model.to('cuda')

root_data = "/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023"

all_map_keyframes = glob.glob(os.path.join(root_data, '**', 'map-keyframes', '*.csv'))
for map_keyframes in tqdm(all_map_keyframes):
    video = os.path.splitext(os.path.basename(map_keyframes))[0]
    asr_feature_path = f'./data/asr_feature/{video}.npy'
    ocr_feature_path = f'./data/ocr_feature/{video}.npy'

    if os.path.exists(asr_feature_path) and os.path.exists(ocr_feature_path):
        continue

    if video == 'L20_V010':
        continue

    df = pd.read_csv(map_keyframes)

    all_keyframes_in_this_video = glob.glob(os.path.join(root_data, '**', 'keyframes', video, '*.jpg'))
    batch_no = map_keyframes[len('/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-')]
    all_keyframes_in_this_video.sort()

    asr_sentences = []
    ocr_sentences = []

    for path in all_keyframes_in_this_video:
        keyframe_position_str = os.path.splitext(os.path.basename(path))[0]
        keyframe_position = int(keyframe_position_str)

        keyframe_idx = df.loc[df['n'] == keyframe_position, 'frame_idx'].values[0]
        pts_time = df.loc[df['n'] == keyframe_position, 'pts_time'].values[0]
        start_time = min(0, pts_time - 10)

        if path == '/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-1/keyframes/L01_V001/0003.jpg':
            keyframe_idx = 271

        keyframe = f"{video}_{keyframe_idx}"
        asr_path = f'/mlcv1/WorkingSpace/Personal/tunglx/AIC23/VideoRetrieval/backend/data/asr_full/{keyframe}.txt'
        ocr_path = f'/mlcv1/WorkingSpace/Personal/thuongpt/OCR/Batch_{batch_no}/{video}/{keyframe_position_str}.txt'

        try:
            with open(asr_path, 'r') as f:
                asr_sentences.append(f.read())
        except Exception as e:
            print(f"Error reading ASR file {asr_path}: {e}")
            asr_sentences.append("")  # Handle the error by appending an empty string

        try:
            with open(ocr_path, 'r') as f:
                ocr_sentences.append(f.read())
        except Exception as e:
            print(f"Error reading OCR file {ocr_path}: {e}")
            ocr_sentences.append("")  # Handle the error by appending an empty string

    # Tokenize and embed ASR sentences
    encoded_input = tokenizer(asr_sentences, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling for ASR
    asr_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().detach().numpy()
    np.save(asr_feature_path, asr_embeddings)
    print(f"ASR embeddings saved for {video}: {asr_embeddings.shape}")

    # Tokenize and embed OCR sentences
    encoded_input = tokenizer(ocr_sentences, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling for OCR
    ocr_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().detach().numpy()
    np.save(ocr_feature_path, ocr_embeddings)
    print(f"OCR embeddings saved for {video}: {ocr_embeddings.shape}")
