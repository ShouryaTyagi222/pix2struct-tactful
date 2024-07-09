# apply he poolings to the embeddings outputss

from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import torch
import os
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("google/pix2struct-docvqa-base")


class Pix2StructDataset(Dataset):
    def __init__(self, data, image_dir, processor, max_patches):
        self.data = data
        self.processor = processor
        self.max_patches = max_patches
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # print(item['file_name'])
            image = Image.open(os.path.join(self.image_dir, item['file_name']))
        except Exception as e:
            print('Error :',e)
            return None
        processed_data = self.processor(images=image, return_tensors="pt", text=item["question"], max_patches=self.max_patches)
        encoding = {}
        for key in processed_data.keys():
            if key in ['flattened_patches', 'attention_mask']:
                encoding[key] = processed_data[key].squeeze()
        encoding['answer'] = item['answer']
        encoding['question'] = item['question']
        encoding['document'] = item['file_name']
        return encoding



class Pix2StructEncoder():
    def __init__(self):
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")
        self.encoder_model = model.encoder
        self.encoder_model.to(device)
        self.encoder_model.eval()

    def get_embeds(self, data_file, image_dir):
        st_time = time.time()
        
        embeds = []
        for item in tqdm(data_file[:3]):
            try:
                # print(item['file_name'])
                image = Image.open(os.path.join(image_dir, item['file_name']))
                # image = image.resize((224, 224))
            except Exception as e:
                print('Error :',e)
                return None
            processed_data = processor(images=image, return_tensors="pt", text=item["question"], max_patches=1024)

            documents = item['file_name']
            flattened_patches = processed_data["flattened_patches"].to(device)
            attention_mask = processed_data["attention_mask"].to(device)

            with torch.no_grad():
                outputs = self.encoder_model(flattened_patches=flattened_patches, attention_mask=attention_mask)
                pooled_output, _ = torch.max(outputs.last_hidden_state, dim=1)
            # print(outputs)
            # print('OUTPUTS KEYS :', outputs.keys())
            # print('ENCODER LAST HIDDEN STATES :', outputs.last_hidden_state)
            # print('ENCODER LAST HIDDEN STATES SHAPE :', pooled_output.shape)
            d_hist = pooled_output.data.cpu().numpy().flatten()
            d_hist /= np.sum(d_hist)
            # print('ENCODER LAST HIDDEN STATES :', d_hist)
            # print('PREPROCESSED ENCODER LAST HIDDEN STATES SHAPE :', d_hist.shape)

            embeds.append({
                'img':  os.path.basename(documents),
                'hist': d_hist
            })
                
        print('completed in :', time.time()-st_time)
        return embeds



if __name__ == '__main__':
    with open('/data/circulars/DATA/pix2struct+tactful/data-1/docvqa_train.json') as f:
        query_data = json.load(f)


    f_model = Pix2StructEncoder()
    print('RUNNING ')
    query_set_embeddings = f_model.get_embeds(
            query_data[:2], '/data/circulars/DATA/pix2struct+tactful/data-1/train')

    print(query_set_embeddings)
    print(len(query_set_embeddings))