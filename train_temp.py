from PIL import Image
import requests
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import wandb

from tqdm import tqdm

# Set the GPU to 1 torch
# processor = AutoProcessor.from_pretrained("google/pix2struct-docvqa-base")
processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")

# Set up the project
# wandb.init(project="pix2struct-synth-base")

# # Wandb config
learning_rate = 1e-5
epochs = 50
batch_size = 12

# Dataset
import json
import os

# with open('final_dataset.json') as f:
#     data_final = json.load(f)

# data_final[0].keys()

from torch.utils.data import Dataset, DataLoader
import cv2

max_patches = 1024

print('RUNNING')

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
        # print(item)
        # Load the image
        # print("Item Keys", item.keys())
        try:
            # print(item['file_name'])
            image = Image.open(os.path.join(self.image_dir, item['file_name']))
        except Exception as e:
            print('Error :',e)
            return None
        processed_data = self.processor(images=image, return_tensors="pt", text=item["question"], max_patches=self.max_patches)
        # print(type(processed_data))
        encoding = {}
        for key in processed_data.keys():
            if key in ['flattened_patches', 'attention_mask']:
                encoding[key] = processed_data[key].squeeze()
                # print(key, processed_data[key])
        encoding['answer'] = item['answer']
        encoding['question'] = item['question']
        encoding['document'] = item['file_name']
        # print("Encoding Keys", encoding.keys())
        return encoding

# import torch
# processor_base = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")
    
def collator(batch):
  # print("Collating")
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  texts = [item["answer"] for item in batch]
  questions = [item["question"] for item in batch]

  documents = [item["document"] for item in batch]
  
  text_inputs = processor(text=texts, padding="max_length", return_tensors="pt", max_length=128, truncation=True)
  
  new_batch["labels"] = text_inputs.input_ids
  
  for item in batch:
    # print("Item Keys", item.keys())
    new_batch["flattened_patches"].append(item["flattened_patches"])
    new_batch["attention_mask"].append(item["attention_mask"])
  
  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"]) 
  new_batch["document"] = documents
  new_batch['questions'] = questions

  return new_batch


data_tr = json.load(open('/data/circulars/DATA/pix2struct+tactful/data-2/docvqa_train.json'))
data_lk = json.load(open('/data/circulars/DATA/pix2struct+tactful/data-2/docvqa_lake.json'))

print('loading the dataloader for train set')
train_dataset = Pix2StructDataset(data_tr, '/data/circulars/DATA/pix2struct+tactful/data-2/train', processor, max_patches=1024)
train_dataset = [item for item in train_dataset if item is not None]
print('dataloading complete')
test_dataset = Pix2StructDataset(data_lk, '/data/circulars/DATA/pix2struct+tactful/data-2/lake', processor, max_patches=1024)
test_dataset = [item for item in test_dataset if item is not None]

# Remove all the None type entries from the dataset
# dataset = [item for item in tqdm(dataset, desc="Removing None entries", unit="item") if item is not None]
print(len(train_dataset))
print(len(test_dataset))
print("Creating train dataloader...")
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator, num_workers=4)
print("Creating test dataloader...")
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator, num_workers=4)

# Training Loop
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()

# Hyperparamters
learning_rate = 1e-5
epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Output examples, write to a seperate file
output_file = open("output_3_base.txt", "w")
