import os
import numpy as np
from tqdm import tqdm
import Levenshtein
import warnings
warnings.filterwarnings("ignore")

from configs import *
from src.scores import cal_scores

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def get_output(batch,model,model_type,device):
    labels = batch["labels"].to(device)
    flattened_patches = batch["flattened_patches"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
    
    return outputs

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
            labels = batch["labels"].to(device)
            documents = batch['document']
            flattened_patches = batch["flattened_patches"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)

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
        with torch.no_grad():
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

                progbar.set_description("Epoch : %s, Test Loss : %0.3f, current Loss : %0.3f" % (epoch+1, np.mean(Loss), loss.item()))

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
        logs = {
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
            }
        scheduler.step()
        epoch+=1

    return model, scheduler, optimizer, logs
      