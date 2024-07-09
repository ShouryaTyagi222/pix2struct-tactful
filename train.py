import time
import os
import torch
import gc

from transformers import AutoProcessor
from transformers import Pix2StructForConditionalGeneration
from transformers import AdamW

import wandb

import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import CosineAnnealingLR

from tactful_exp.tactful_smi import TACTFUL_SMI
from src.helper import *
from configs import *
from src.run import run_a_round
from utils.data import load_data

print('RUNNING')  
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
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
            image = Image.open(os.path.join(self.image_dir, item['file_name'])).resize((IMAGE_WIDTH,IMAGE_HEIGHT))
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


def main():
    iteration = args['iterations']
    selection_strag = args['strategy']
    budget = args['proposal_budget']

    # LOG IN TO WANDB
    if wandb_flag:
        wandb.init(project=wandb_project_desc, name=wandb_name)
        wandb.login(key=wandb_key)
        wandb.config.update({"learning_rate": learning_rate, "batch_size": args['batch_size'], "num_epochs": iteration, "model": wandb_model_desc})

    print('STRATEGY :', selection_strag)

    # LOADING INITIAL REQUIREMENTS
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)
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
    i = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{args['device']}")
    with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
        f.write("")

    # print('dataloading started')
    # t1 = time.time()
    # test_file = json.load(open(val_data_dirs[1]))
    # test_dataset = Pix2StructDataset(test_file, val_data_dirs[0], processor, max_patches=1024)
    # print('Data Size :', len(test_dataset))
    # test_dataset = [item for item in test_dataset if item is not None]
    # test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args['batch_size'], collate_fn=collator, num_workers = 4, pin_memory = True)
    # print('Data Loaded in :', time.time() - t1)
    test_dataloader=load_data(val_data_dirs[1], val_data_dirs[0], processor, args['batch_size'])

    # INITIAL TRAINING
    if not os.path.exists(os.path.join(output_dir,'model')):
        print('Loading the Initial Model')

        # INITIALIZING THE MODEL AND OPTIMIZER
        model = Pix2StructForConditionalGeneration.from_pretrained(INIT_MODEL_PATH)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)
        model.to(device)


        # LOADING THE DATALOADER
        # print('dataloading started')
        # t1 = time.time()
        # train_file = json.load(open(train_data_dirs[1]))
        # train_dataset = Pix2StructDataset(train_file, train_data_dirs[0], processor, max_patches=1024)
        # print('Data Size :', len(train_dataset))
        # train_dataset = [item for item in train_dataset if item is not None]
        # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'], collate_fn=collator, num_workers = 4, pin_memory = True)
        # print('Data Loaded in :', time.time() - t1)
        train_dataloader=load_data(train_data_dirs[1], train_data_dirs[0], processor, args['batch_size'])

        model.train()
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write("\nStarting the Initial Training\n\n")

        torch.cuda.empty_cache()

        print('Training the Initial Model')
        st = time.time()
        model, scheduler, optimizer, logs = run_a_round(train_dataloader,test_dataloader,scheduler,optimizer,model,device,wandb_flag,iteration,processor)
        et = time.time()
        print('TIME TAKES :', et-st)

        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f"TIME TAKEN : {et-st}\n")

        if wandb_flag:
            wandb.log(logs)

        model.save_pretrained(os.path.join(output_dir,'model'))
        print('saved the models in :', os.path.join(output_dir,'model'))

        t_flag = 1
        del train_dataloader
        gc.collect()

    else:
        print('Loading the Saved Model')
        try:
            model = Pix2StructForConditionalGeneration.from_pretrained(INIT_MODEL_PATH)
            model.to(device)
        except:
            print('The Saved model is not found.')
            return
        t_flag = 0 

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)
    # STARTING THE AL ROUND

    i=0

    while (i < iteration and len(os.listdir(lake_data_dirs[0])) > 0):
        st = time.time()

        print(f'AL Round : {i}/{iteration}')
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f'\nAL Round : {i}/{iteration}\n')

        # SELECTING THE DATA USING TACTFUL
        if t_flag:
            if (selection_strag != "random"):
                print('DOING TACTFUL')
                selection_arg['iteration'] = i
                strategy_sel = TACTFUL_SMI(args = selection_arg)
                lake_image_list, subset_result = strategy_sel.select(budget, train_data_dirs[1], lake_data_dirs[1], train_data_dirs[0], lake_data_dirs[0], processor)
                print('LENGTH OF SUBSET RESULT :',subset_result)
                subset_result = [lake_image_list[i] for i in subset_result]
                subset_result = list(
                    set(subset_result))
                print(subset_result)
                print(len(subset_result))

            else:
                lake_image_list = os.listdir(lake_data_dirs[0])
                subset_result = Random_wrapper(
                    lake_image_list, budget)

            # REDUCING THE BUDGET SIZE
            print('BUDGET LEFT:', len(os.listdir(lake_data_dirs[0])))

            # TRANSFERRING THE DATA FROM LAKE SET TO THE TRAIN SET
            if (len(os.listdir(lake_data_dirs[0])) > 0):
                data_len = aug_train_subset(subset_result, train_data_dirs[1], lake_data_dirs[1], lake_data_dirs, train_data_dirs)
                with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                    f.write(f'LENGTH OF THE TRAIN DATA : {data_len}\n')
                print('LENGTH OF THE TRAIN DATA : ',data_len)
        t_flag = 1
           
        torch.cuda.empty_cache()
        # LOADING THE NEW DATALOADER
        # print('dataloading started')
        # t1 = time.time()
        # train_file = json.load(open(train_data_dirs[1]))
        # train_dataset = Pix2StructDataset(train_file, train_data_dirs[0], processor, max_patches=1024)
        # print('Data Size :', len(train_dataset))
        # train_dataset = [item for item in train_dataset if item is not None]
        # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'], collate_fn=collator, num_workers = 4, pin_memory = True)
        # print('Data Loaded in :', time.time() - t1)
        train_dataloader=load_data(train_data_dirs[1], train_data_dirs[0], processor, args['batch_size'])

        model, scheduler, optimizer, logs = run_a_round(train_dataloader,test_dataloader,scheduler,optimizer,model,device,wandb_flag,iteration, processor)

        et = time.time()
        print('TIME TAKEN :', et-st)
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f"TIME TAKEN : {et-st}\n")

        if wandb_flag:
            wandb.log(logs)

        torch.cuda.empty_cache()
        del train_dataloader
        gc.collect()

        i += 1

    model.save_pretrained(os.path.join(output_dir,'model'))


if __name__ == "__main__":
    main()