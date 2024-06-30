import os
import torch

from transformers import AutoProcessor
from transformers import Pix2StructForConditionalGeneration
from transformers import AdamW

import numpy as np
import wandb

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import CosineAnnealingLR

from tactful_exp.tactful_smi import TACTFUL_SMI
from src.helper import *
from configs import *

print('RUNNING')

def run_a_round(train_dataloader,test_dataloader,scheduler,optimizer,model,device,wandb_flag,iteration, processor):
    epoch=0
    rougel_f_tr=0
    prev_train_loss = 0
    train_loss = 1
    temp_train_loss = 1
    sat_epoch = 0
    print('T_max :',iteration*MAX_AL_EPOCHS)

    while (rougel_f_tr<=ROUGE_THRESH and sat_epoch < SATURATION_THRESH and train_loss>=TRAIN_LOSS_THRESH) or epoch<EPOCH_THRESH and epoch<MAX_AL_EPOCHS:
        # RUNNING EPOCH
        model.train()
        progbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}, Train Loss = 0, current loss = 0 , Bleu Score = 0', unit='batch')
        Loss=[]
        print(f'Epoch : {epoch+1}, learning rate : {optimizer.param_groups[0]["lr"]}')

        # TRAINING
        for batch in progbar:
            labels = batch["labels"]
            documents = batch['document']

            outputs = get_output(batch,model,model_name,device)

            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            Loss.append(loss.item())
            progbar.set_description("Epoch : %s, Train Loss : %0.3f, current Loss : %0.3f" % (epoch+1, np.mean(Loss), loss.item()))

        # CALCULATING SCORES
        rouge1_f_tr, rougel_f_tr, anls_tr = cal_scores(outputs, labels, documents, processor)
        print(f'rouge1_f1 : {rouge1_f_tr}, rougeL_f1 : {rougel_f_tr}, anls : {anls_tr},\n')

        # SAVING THE SCORES
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write("\nEpoch = %s Train Loss = %0.3f, Learning Rate = %s \n" % (epoch+1, np.mean(Loss), optimizer.param_groups[0]["lr"]))
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f'rouge1_f1 : {rouge1_f_tr}, rougeL_f1 : {rougel_f_tr}, anls : {anls_tr},\n')
        print(f'Epoch : {epoch+1}, learning rate : {optimizer.param_groups[0]["lr"]}')

        # SETTING CHECKS FOR NEXT LOOP
        prev_train_loss = train_loss 
        if abs(round(np.mean(Loss),2)-round(prev_train_loss,2))<=0.05:
            sat_epoch+=1
        else:
            sat_epoch = 0
            train_loss=np.mean(Loss)
        temp_train_loss=np.mean(Loss)

        # MODEL EVALUATION
        model.eval()
        Loss = []
        total_rouge1 = 0
        total_rougeL = 0
        total_samples = 0
        total_anls = 0
        progbar = tqdm(test_dataloader, desc=f'Epoch {epoch}', unit='batch')
        for batch in progbar:
            labels = batch["labels"]
            documents = batch['document']

            outputs = get_output(batch,model,model_name,device)

            loss = outputs.loss
            Loss.append(loss.item())

            # Decode predictions and labels
            predictions = processor.batch_decode(outputs.logits.argmax(-1))
            labels = processor.batch_decode(labels)

            # Calculate ROUGE scores
            for pred, label in zip(predictions, labels):
                try:
                    eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
                    eos_label = label.index("</s>") if "</s>" in label else len(label)

                    pred_no_pad = [token for token in pred[:eos_pred] if token != '<pad>']
                    label_no_pad = [token for token in label[:eos_label] if token != '<pad>']

                    scores = scorer.score(" ".join(pred_no_pad), " ".join(label_no_pad))
                    total_rouge1 += scores['rouge1'].fmeasure
                    total_rougeL += scores['rougeL'].fmeasure
                    total_samples += 1

                    dist = Levenshtein.distance(" ".join(pred_no_pad), " ".join(label_no_pad))
                    normalized_dist = dist / len(label[:eos_label])
                    total_anls = total_anls + normalized_dist
                except :
                    # print('Error in Score Calculation')
                    pass

            with open(os.path.join(model_path,f"{model_name}_sample_q_and_a.txt"), 'a') as f:
                for doc, pred, label in zip(documents, predictions, labels):
                    eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
                    eos_label = label.index("</s>") if "</s>" in label else len(label)

                    f.write("Document: " + str(doc) + "\n")
                    f.write("Predictions: " + str(pred[:eos_pred]) + "\n")
                    f.write("Labels: " + str(label[:eos_label]) + "\n")

            progbar.set_description("Epoch : %s, Train Loss : %0.3f, current Loss : %0.3f" % (epoch+1, np.mean(Loss), loss.item()))

        # SAVING SCORES 
        avg_anls = total_anls / total_samples  
        avg_rouge1 = total_rouge1 / total_samples
        avg_rougeL = total_rougeL / total_samples            
           
        print(f'rouge1_f1 : {avg_rouge1}, rougeL_f1 : {avg_rougeL}, anls : {avg_anls},\n')

        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write("Epoch = %s Test Loss = %0.3f, Learning Rate = %s \n" % (epoch+1, np.mean(Loss), optimizer.param_groups[0]["lr"]))
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f'rouge1_f1 : {avg_rouge1}, rougeL_f1 : {avg_rougeL}, anls : {avg_anls},\n')

        # SAVING DATA IN WandB
        if wandb_flag:
            wandb.log({
                "Training Loss": temp_train_loss,
                "Testing Loss": np.mean(Loss),
                "Epoch": epoch,
                "Data Size": len(train_dataloader),
                "Train Rouge-1": rouge1_f_tr,
                "Train Rouge-L": rougel_f_tr,
                "Train ANLS Score": anls_tr,
                "Test Rouge-1": avg_rouge1,
                "Test Rouge-L": avg_rougeL,
                "Test ANLS": avg_anls,
            })
        scheduler.step()
        epoch+=1

    return model, scheduler, optimizer
        
        


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
        model, scheduler, optimizer = run_a_round(train_dataloader,test_dataloader,scheduler,optimizer,model,device,wandb_flag,iteration,processor)
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
                set(get_original_images_path(subset_result,lake_data_dirs[0])))
            print(subset_result)

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

        model, scheduler, optimizer = run_a_round(train_dataloader,test_dataloader,scheduler,optimizer,model,device,wandb_flag,iteration, processor)
        torch.cuda.empty_cache()
        i += 1
        print("remaining_budget", budget)

    model.save_pretrained(os.path.join(output_dir,'model'))


if __name__ == "__main__":
    main()