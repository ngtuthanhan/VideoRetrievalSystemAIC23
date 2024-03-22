import subprocess
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import pandas as pd
import librosa
import os
import glob
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model.to(device)

def transcribe(wav):
    input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.batch_decode(pred_ids)[0]
    return pred_transcript

def process_audio_in_segments(segments_df, video_path, root_folder='/mlcv1/WorkingSpace/Personal/tunglx/AIC23/VideoRetrieval/backend/data/asr_full'):
    # Initialize an empty list to store the data for the CSV file
    lst = range(len(segments_df))
    for i in lst:
        frame_idx = segments_df['frame_idx'][i]
        video = video_path.split('/')[-1].replace('.mp4', '')
        frame_idx = str(frame_idx)
        keyframe = video + "_" + frame_idx
        save_file = os.path.join(root_folder, keyframe) + '.txt'
        
        if os.path.exists(save_file):
            continue
        # if not pd.isna(pts_time):
        start_time = segments_df['pts_time'][i-1]  if i>0 else 0
        end_time = segments_df['pts_time'][i+1]  if i<len(segments_df)-1 else None
        audio_filename = 'temp_audio_4_3.wav'
    
        cmd = f'ffmpeg -i "{video_path}" -ss {start_time} -to {end_time} -vn -acodec pcm_s16le -ar 16000 -ac 2 "{audio_filename}"  -v 0' \
              if end_time != None else f'ffmpeg -i "{video_path}" -ss {start_time} -vn -acodec pcm_s16le -ar 16000 -ac 2 "{audio_filename}" -v 0'
        subprocess.run(cmd, shell=True, check=True)
        segment, _ = librosa.load(audio_filename, sr=16000)
        os.remove(audio_filename)

        # Replace this with your actual transcription function
        transcription = transcribe(segment)
        with open(save_file, 'w') as f:
            f.write(transcription)
        


# all_videos = glob.glob("/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/**/video/*.mp4",recursive=True)
batch_videos = glob.glob("/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-3/video/*.mp4")
for video_path in tqdm(batch_videos):
    if '/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-2/video/L20_V010.mp4' == video_path:
        continue
    map_path =  video_path.replace('video', 'map-keyframes').replace('.mp4', '.csv')
    segments_df = pd.read_csv(map_path)
    # Process audio based on segment timestamps
    process_audio_in_segments(segments_df, video_path)

    print("Transcription process complete.")