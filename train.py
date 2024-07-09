import os
import torch

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

def main():
    iteration = args['iterations']
    selection_strag = args['strategy']
    selection_budget = args['budget']
    budget = args['total_budget']
    proposal_budget = args['proposal_budget']

    # LOG IN TO WANDB
    if wandb_flag:
        wandb.init(project=wandb_project_desc, name=wandb_name)
        wandb.login(key=wandb_key)
        wandb.config.update({"learning_rate": learning_rate, "batch_size": args['batch_size'], "num_epochs": iteration, "model": wandb_model_desc})
        wandb.login()

    print('STRATEGY :', selection_strag)

    # LOADING INITIAL REQUIREMENTS
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)   
    i = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{args['device']}")
    with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
        f.write("")
    test_dataloader=load_data(val_data_dirs[1], val_data_dirs[0], processor, args['batch_size'])
    print('test dataloader loaded')

    # INITIAL TRAINING
    if not os.path.exists(os.path.join(output_dir,'model')):
        print('Loading the Initial Model')

        # INITIALIZING THE MODEL AND OPTIMIZER
        model = Pix2StructForConditionalGeneration.from_pretrained(INIT_MODEL_PATH)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)
        model.to(device)

        # LOADING THE DATALOADER
        print('dataloading started')
        train_dataloader=load_data(train_data_dirs[1], train_data_dirs[0], processor, args['batch_size'])
        print('train dataloader loaded')

        torch.cuda.empty_cache()
        model.train()
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write("\nStarting the Initial Training\n\n")

        print('Training the Initial Model')
        model, scheduler, optimizer, logs = run_a_round(train_dataloader,test_dataloader,scheduler,optimizer,model,device,wandb_flag,iteration,processor)
        if wandb_flag:
            wandb.log(logs)
        model.save_pretrained(os.path.join(output_dir,'model'))
        print('saved the models in :', os.path.join(output_dir,'model'))

    else:
        print('Loading the Saved Model')
        try:
            model = Pix2StructForConditionalGeneration.from_pretrained(INIT_MODEL_PATH)
            model.to(device)
        except:
            print('The Saved model is not found.')
            return 

    # STARTING THE AL ROUND

    i=0
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)

    while (i < iteration and budget > 0):

        print(f'AL Round : {i}/{iteration}')
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f'\nAL Round : {i}/{iteration}\n')

        # SELECTING THE DATA USING TACTFUL
        if (selection_strag != "random"):
            print('DOING TACTFUL')
            selection_arg['iteration'] = i
            strategy_sel = TACTFUL_SMI(args = selection_arg)
            lake_image_list, subset_result = strategy_sel.select(proposal_budget, train_data_dirs[1], lake_data_dirs[1], train_data_dirs[0], lake_data_dirs[0], processor)
            print('LENGTH OF SUBSET RESULT :',subset_result)
            subset_result = [lake_image_list[i] for i in subset_result]
            subset_result = list(
                set(subset_result))
            print(subset_result)
            print(len(subset_result))

        else:
            lake_image_list = os.listdir(lake_data_dirs[0])
            subset_result = Random_wrapper(
                lake_image_list, selection_budget)

        # REDUCING THE BUDGET SIZE
        budget -= len(subset_result)

        # TRANSFERRING THE DATA FROM LAKE SET TO THE TRAIN SET
        if (budget > 0):
            aug_train_subset(subset_result, train_data_dirs[1], lake_data_dirs[1], budget, lake_data_dirs, train_data_dirs)
            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write(f'LENGTH OF THE TRAIN DATA : {len(os.listdir(train_data_dirs[0]))}\n')
            print('LENGTH OF THE TRAIN DATA : ',len(os.listdir(train_data_dirs[0])))
           
        # LOADING THE NEW DATALOADER
        train_dataloader=load_data(train_data_dirs[1], train_data_dirs[0], processor, args['batch_size'])

        model, scheduler, optimizer, logs = run_a_round(train_dataloader,test_dataloader,scheduler,optimizer,model,device,wandb_flag,iteration, processor)
        if wandb_flag:
            wandb.log(logs)
        torch.cuda.empty_cache()
        i += 1
        print("remaining_budget", budget)

    model.save_pretrained(os.path.join(output_dir,'model'))


if __name__ == "__main__":
    main()