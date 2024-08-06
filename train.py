import time
import os
import torch
import gc

from transformers import AutoProcessor
from transformers import Pix2StructForConditionalGeneration
from transformers import AdamW

import wandb
from torch.utils.tensorboard import SummaryWriter

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
    budget = args['proposal_budget']

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(model_path,'tensorboard_logs'))

    # LOG IN TO WANDB
    if wandb_flag:
        wandb.init(project=wandb_project_desc, name=wandb_name)
        wandb.login(key=wandb_key)
        wandb.config.update({"learning_rate": learning_rate, "batch_size": args['batch_size'], "num_epochs": iteration, "model": wandb_model_desc})

    print('STRATEGY :', selection_strag)

    # LOADING INITIAL REQUIREMENTS
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)

    i = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{args['device']}")
    with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
        f.write("")

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
        train_dataloader=load_data(train_data_dirs[1], train_data_dirs[0], processor, args['batch_size'])

        model.train()
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write("\nStarting the Initial Training\n\n")

        torch.cuda.empty_cache()

        print('Training the Initial Model')
        st = time.time()
        model, scheduler, optimizer, logs = run_a_round(train_dataloader, test_dataloader, scheduler, optimizer, model, device, wandb_flag, iteration, processor)
        et = time.time()
        print('TIME TAKEN :', et-st)

        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f"TIME TAKEN : {et-st}\n")

        if wandb_flag:
            wandb.log(logs)
        writer.add_scalar('Training Loss', logs['Training Loss'], 0)
        writer.add_scalar('Testing Loss', logs['Testing Loss'], 0)
        writer.add_scalar('Train Rouge-1', logs['Train Rouge-1'], 0)
        writer.add_scalar('Train Rouge-L', logs['Train Rouge-L'], 0)
        writer.add_scalar('Train ANLS Score', logs['Train ANLS Score'], 0)
        writer.add_scalar('Test Rouge-1', logs['Test Rouge-1'], 0)
        writer.add_scalar('Test Rouge-L', logs['Test Rouge-L'], 0)
        writer.add_scalar('Test ANLS', logs['Test ANLS'], 0)

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
    # STARTING THE AL ROUND

    i=0

    while (i < iteration and len(os.listdir(lake_data_dirs[0])) > 0):
        st = time.time()

        print(f'AL Round : {i}/{iteration}')
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f'\nAL Round : {i}/{iteration}\n')

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)

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
        train_dataloader=load_data(train_data_dirs[1], train_data_dirs[0], processor, args['batch_size'])

        model, scheduler, optimizer, logs = run_a_round(train_dataloader, test_dataloader, scheduler, optimizer, model, device, wandb_flag, iteration, processor)

        et = time.time()
        print('TIME TAKEN :', et-st)
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f"TIME TAKEN : {et-st}\n")

        if wandb_flag:
            wandb.log(logs)
        writer.add_scalar('Training Loss', logs['Training Loss'], i+1)
        writer.add_scalar('Testing Loss', logs['Testing Loss'], i+1)
        writer.add_scalar('Train Rouge-1', logs['Train Rouge-1'], i+1)
        writer.add_scalar('Train Rouge-L', logs['Train Rouge-L'], i+1)
        writer.add_scalar('Train ANLS Score', logs['Train ANLS Score'], i+1)
        writer.add_scalar('Test Rouge-1', logs['Test Rouge-1'], i+1)
        writer.add_scalar('Test Rouge-L', logs['Test Rouge-L'], i+1)
        writer.add_scalar('Test ANLS', logs['Test ANLS'], i+1)

        torch.cuda.empty_cache()
        del train_dataloader
        gc.collect()

        i += 1

    model.save_pretrained(os.path.join(output_dir,'model'))
    writer.close()

if __name__ == "__main__":
    main()
