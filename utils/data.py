import os, json
import torch
from PIL import Image
import sys
sys.path.append("../")
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
import time

import sys
sys.path.append("../")



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

def load_data(input_file, image_dir, processor, batch_size):

    print('Loading the Data')
    t1 = time.time()
    data_file = json.load(open(input_file))
    print('Data Size :', len(data_file))
    
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

    dataset = Pix2StructDataset(data_file, image_dir, processor, max_patches=1024)
    # dataset = Pix2StructDataset1(data_file)

    dataset = [item for item in dataset if item is not None]
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collator)
    # dataset = [item for item in dataset if item is not None]
    # dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collator1)
    print('Data Loaded in :', time.time() - t1)

    return dataloader
