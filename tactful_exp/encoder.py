# calculates the embeddings of image-question pairs

from PIL import Image
import requests
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import json
import torch
from sklearn.model_selection import train_test_split
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
    

def collator(batch):
  # print("Collating")
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  documents = [item["document"] for item in batch]

  for item in batch:
    # print("Item Keys", item.keys())
    new_batch["flattened_patches"].append(item["flattened_patches"])
    new_batch["attention_mask"].append(item["attention_mask"])
  
  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"]) 
  new_batch["document"] = documents

  return new_batch


class Pix2StructEncoder():
    def __init__(self):
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")
        self.encoder_model = model.encoder
        self.encoder_model.to(device)
        self.encoder_model.eval()

    def get_embeds(self, data_file, image_dir):
        st_time = time.time()
        print("Data Generation Starting...")
        dataset = Pix2StructDataset(data_file, image_dir, processor, max_patches=1024)
        dataset = [item for item in dataset if item is not None]
        print('data set generated')
        dataloader = DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=collator, num_workers=4)

        with torch.no_grad():
            embeds = []
            for idx_1, batch_1 in enumerate(tqdm(dataloader)):
                documents = batch_1["document"]
                flattened_patches = batch_1["flattened_patches"].to(device)
                attention_mask = batch_1["attention_mask"].to(device)

                outputs = self.encoder_model(flattened_patches=flattened_patches, attention_mask=attention_mask)
                print(outputs)
                print('OUTPUTS KEYS :', outputs.keys())
                print('ENCODER LAST HIDDEN STATES :', outputs.last_hidden_state)
                print('ENCODER LAST HIDDEN STATES SHAPE :', outputs.last_hidden_state.shape)
                d_hist = outputs.last_hidden_state.data.cpu().numpy().flatten()
                d_hist /= np.sum(d_hist)
                print('ENCODER LAST HIDDEN STATES :', d_hist)
                print('ENCODER LAST HIDDEN STATES SHAPE :', d_hist.shape)

                embeds.append({
                    'img':  os.path.basename(documents[0]),
                    'hist': d_hist
                })
                
        print('completed in :', time.time()-st_time)
        return embeds



if __name__ == '__main__':
    with open('/data/circulars/DATA/pix2struct+tactful/data-1/docvqa_train.json') as f:
        query_data = json.load(f)


    f_model = Pix2StructEncoder()
    query_set_embeddings = f_model.get_embeds(
            query_data, '/data/circulars/DATA/pix2struct+tactful/data-1/train')

    print(query_set_embeddings)
    print(len(query_set_embeddings))