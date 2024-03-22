import glob
import numpy as np
import os
from tqdm import tqdm 
import pandas as pd
class Indexing:
    def __init__(self,root_data):
        self.root_data = root_data
        self.all_data_features = glob.glob(os.path.join(root_data,'**','clip-features-vit-b32', '*.npy')) 
        self.all_data_features += glob.glob(os.path.join(root_data,'**','clip-features-32', '*.npy'))
        self.all_map_keyframes = glob.glob(os.path.join(root_data,'**','map-keyframes', '*.csv'))
        self.all_data_images = glob.glob(os.path.join(root_data,'**','keyframes', '**', '*.jpg'))

    @property
    def get_all_features_from_files(self):
        features = np.empty([0,512],dtype = float)
        index = []
        for folder_keyframes in tqdm(self.all_data_features):
            # Add features vector
            feature_video = np.load(folder_keyframes)
            # feature_video /= np.linalg.norm(feature_video, axis=-1, keepdims=True)
            features = np.concatenate((features,feature_video),axis = 0)
            map_keyframes = folder_keyframes.replace('.npy','.csv').replace('clip-features-32','map-keyframes').replace('clip-features-vit-b32','map-keyframes')
            df = pd.read_csv(map_keyframes)
            id_video = list(df['frame_idx'])
            if folder_keyframes == '/mlcv1/Datasets/HCM_AIChallenge/HCM_AIC_2023/data-batch-1/clip-features/L01_V001.npy':
                id_video[2] = 271
            video = folder_keyframes.split('/')[-1].replace('.npy', '')
            for id in id_video:
                index.append(str(video)+"_"+str(id))
        return features,np.array(index)
    
    # @property
    # def get_all_features(self):
    #     features = np.empty([0,512],dtype = float)
    #     index = []
    #     model_name = 'ViT-B/32'
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     model, preprocess = clip.load(model_name, device=device)
    #     for image in tqdm(self.all_data_images):
    #         video, frame = image.split('_')[-2].split('/')[-1], image.split('_')[-1]
            
    #         image_input = self.preprocess(image).unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             image_features = model.encode_image(image_input)
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #         features = np.concatenate((features,image_features.cpu().detach().numpy()),axis = 0)
    #         index.append(str(video)+"_"+str(id))
    #     return features, np.array(index)