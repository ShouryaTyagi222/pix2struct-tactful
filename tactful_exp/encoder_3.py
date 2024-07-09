from PIL import Image
from transformers import Pix2StructForConditionalGeneration
from transformers import AutoProcessor
import numpy as np
from tqdm import tqdm
import json
import torch
import torch
import os
import time
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Pix2StructEncoder():
    def __init__(self, processor):
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")
        self.encoder_model = model.encoder
        self.encoder_model.to(device)
        self.encoder_model.eval()

        self.linear_layer = nn.Linear(786432, 2024).to(device)
        self.processor = processor

    def get_embeds(self, data_file, image_dir):
        st_time = time.time()
        
        embeds = []
        for item in tqdm(data_file):
            try:
                # print(item['file_name'])
                image = Image.open(os.path.join(image_dir, item['file_name']))
                image = image.resize((224, 224))
            except Exception as e:
                print('Error :',e)
                return None
            processed_data = self.processor(images=image, return_tensors="pt", text=item["question"], max_patches=1024)

            documents = item['file_name']
            flattened_patches = processed_data["flattened_patches"].to(device)
            attention_mask = processed_data["attention_mask"].to(device)

            with torch.no_grad():
                outputs = self.encoder_model(flattened_patches=flattened_patches, attention_mask=attention_mask)
                encoded_output = outputs.last_hidden_state.view(outputs.last_hidden_state.size(0), -1)  # Flatten
                encoded_output = encoded_output.to(device)
                
                # Pass through linear layer for reshaping
                output = self.linear_layer(encoded_output)
            d_hist = output.data.cpu().numpy().flatten()
            d_hist /= np.sum(d_hist)
            
            embeds.append({
                'img':  os.path.basename(documents),
                'hist': d_hist
            })
                
        print('completed in :', time.time()-st_time)
        return embeds



if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")   
    with open('/data/circulars/DATA/pix2struct+tactful/data-1/docvqa_train.json') as f:
        query_data = json.load(f)


    f_model = Pix2StructEncoder(processor)
    print('RUNNING ')
    query_set_embeddings = f_model.get_embeds(
            query_data, '/data/circulars/DATA/pix2struct+tactful/data-1/train')

    # print(query_set_embeddings)
    print(len(query_set_embeddings))