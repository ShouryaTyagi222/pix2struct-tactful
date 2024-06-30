import os, shutil, json ,cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import Features, Sequence, Value, Array2D, Array3D
from torch.utils.data.dataloader import default_collate
import sys
sys.path.append("../")
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
from transformers import AutoProcessor
from detectron2.engine import DefaultPredictor
import Levenshtein
import time

import sys
sys.path.append("../")

from configs import *



def create_model(cfg):
    tester = DefaultPredictor(cfg)
    return tester
    

def cal_scores(outputs, labels, documents, processor_base):
    total_rouge1=0
    total_rougeL=0
    total_anls=0
    total_samples = 0

    # Decode predictions and labels
    predictions = processor_base.batch_decode(outputs.logits.argmax(-1))
    labels = processor_base.batch_decode(labels)

    # Calculate ROUGE scores
    for pred, label in zip(predictions, labels):
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
    
    # Write predictions and labels to file
    with open(os.path.join(model_path,f"{model_name}_sample_q_and_a.txt"), 'a') as f:
        for doc, pred, label in zip(documents, predictions, labels):
            eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
            eos_label = label.index("</s>") if "</s>" in label else len(label)

            f.write("Document: " + str(doc) + "\n")
            f.write("Predictions: " + str(pred[:eos_pred]) + "\n")
            f.write("Labels: " + str(label[:eos_label]) + "\n")
    
    avg_anls = total_anls / total_samples  
    avg_rouge1 = total_rouge1 / total_samples
    avg_rougeL = total_rougeL / total_samples

    return avg_rouge1, avg_rougeL, avg_anls

def get_output(batch,model,model_type,device):
    labels = batch["labels"].to(device)
    flattened_patches = batch["flattened_patches"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
    
    return outputs
    
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

# class Pix2StructDataset1(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         # Convert lists back to tensors
#         flattened_patches = torch.tensor(item['flattened_patches'])
#         attention_mask = torch.tensor(item['attention_mask'])
#         labels = torch.tensor(item['labels'])
#         question = item['question']
#         document = item['document']
#         # print(idx, 'Done')
        
#         return {
#             'flattened_patches': flattened_patches,
#             'attention_mask': attention_mask,
#             'labels': labels,
#             'question': question,
#             'document': document
#         }
    

# def collator1(batch):
#   # print("Collating")
#   new_batch = {"flattened_patches":[], "attention_mask":[], 'labels': []}
#   questions = [item["question"] for item in batch]
#   documents = [item["document"] for item in batch]
  
#   for item in batch:
#     # print("Item Keys", item.keys())
#     new_batch["flattened_patches"].append(item["flattened_patches"])
#     new_batch["attention_mask"].append(item["attention_mask"])
#     new_batch['labels'].append(item['labels'])
  
#   new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
#   new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"]) 
#   new_batch["labels"] = torch.stack(new_batch["labels"]) 
#   new_batch["document"] = documents
#   new_batch['questions'] = questions

#   return new_batch


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

def crop_object(image, box, ground_truth=False):
    """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
    """
    if (not ground_truth):
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
    else:
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[0] + box[2]
        y_bottom_right = box[1] + box[3]
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2

    try:
        crop_img = image.crop((int(x_top_left), int(y_top_left),
                               int(x_bottom_right), int(y_bottom_right)))
    except Exception as e:
        pass

    return crop_img

# '''
# Returns the list of cropped images based on the objects. The method make use of ground truth to crop the image.
# '''
# def crop_images_classwise_ground_truth(train_json_path, src_path, dest_path,
#                                        category: str):
#     category=category.lower()
#     if not os.path.exists(dest_path + '/obj_images'):
#         os.makedirs(dest_path + '/obj_images')
#     obj_im_dir = dest_path + '/obj_images'
#     no_of_objects = 0
#     with open(train_json_path) as f:
#         data = json.load(f)
#     file_names = os.listdir(src_path)
    
#     for annot in tqdm(data):
#         img_name = annot['file_name']
#         # print(img_name)
#         if img_name in file_names:
#             img_annots=annot['annotations']
#             img=cv2.imread(os.path.join(src_path,img_name))
#             img=cv2.resize(img,(224,224))
#             # print(img_name,':',img_annots)
#             for i,img_annot in enumerate(img_annots):
#                 if img_annot['label'].lower()==category.lower():
#                     no_of_objects += 1
#                     x,y,w,h=int(img_annot['x']),int(img_annot['y']),int(img_annot['w']),int(img_annot['h'])
#                     crp_img=img[y:y+h,x:x+w]
#                     if y+h<224 and x+w<224:
#                         cv2.imwrite(os.path.join(obj_im_dir,category,'.'.join(img_name.split('.')[:-1])+'_'+str(i)+'.png'),crp_img)
                        

# def crop_images_classwise(src_path, dest_path,
#                           proposal_budget: int):
#     if not os.path.exists(dest_path + '/obj_images'):
#         os.makedirs(dest_path + '/obj_images')
#     model = create_model(cfg)
#     obj_im_dir = dest_path + '/obj_images'
#     no_of_objects = 0
#     print(src_path)
#     for d in tqdm(os.listdir(src_path)):
#         image = cv2.imread(os.path.join(src_path, d))
#         height, width = image.shape[:2]
#         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
#         inputs = [{"image": image, "height": height, "width": width}]
#         images = model.model.preprocess_image(inputs)

#         features = model.model.backbone(images.tensor)
#         proposals, _ = model.model.proposal_generator(images, features)
#         instances, _ = model.model.roi_heads(images, features,
#                                                      proposals)
#         boxes = instances[0].pred_boxes
#         classes = instances[0].pred_classes.cpu().numpy().tolist()
#         max_score_order = torch.argsort(instances[0].scores).tolist()

#         if (proposal_budget > len(max_score_order)):
#             proposal_budget = len(max_score_order)

#         for singleclass in classes:
#             if not os.path.exists(
#                     os.path.join(dest_path, 'obj_images',
#                                  R_MAPPING[str(singleclass)])):
#                 os.makedirs(
#                     os.path.join(dest_path, 'obj_images',
#                                  R_MAPPING[str(singleclass)]))

#         img = Image.open(os.path.join(src_path, d))
#         for idx, box in enumerate(
#                 list(boxes[max_score_order[:proposal_budget]])):
#             no_of_objects += 1
#             box = box.detach().cpu().numpy()

#             crop_img = crop_object(img, box)
#             try:
#                 crop_img.save(
#                     os.path.join(
#                         obj_im_dir, R_MAPPING[str(classes[idx])],
#                         d.replace(".png", "") + "_" + str(idx) + ".png"))
#             except Exception as e:
#                 print(e)

#     print("Number of objects: " + str(no_of_objects))

def Random_wrapper(image_list, budget=10):
    rand_idx = np.random.permutation(len(image_list))[:budget]
    rand_idx = rand_idx.tolist()
    Random_results = [image_list[i] for i in rand_idx]

    return Random_results

def change_dir(image_results, src_dir, dest_dir):
    for image in image_results:
        source_img = os.path.join(src_dir[0],image)
        destination_img = os.path.join(dest_dir[0], os.path.basename(image))
        if not os.path.exists(dest_dir[0]) or not os.path.exists(dest_dir[1]):
            os.mkdir(dest_dir[0])
            os.mkdir(dest_dir[1])

        try:
            shutil.copy(source_img, destination_img)
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")

        # For other errors
        except Exception as e:
            print("Error occurred while copying file.", e)


        # removing the data from the lake data
        try:
            os.remove(source_img)
        except:
            pass

def remove_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except:
        pass

def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass

def get_original_images_path(subset_result:list,img_dir:str):
    return ["_".join(os.path.basename(x).split("_")[:-1])+'.png' for x in subset_result]

def aug_train_subset(subset_result, train_data_json, lake_data_json, budget, src_dir, dest_dir):
    with open(lake_data_json, mode="r") as f:
        lake_dataset = json.load(f)
    with open(train_data_json, mode="r") as f:
        train_dataset = json.load(f)

    new_lake_data=[]

    for data in lake_dataset:
        if data['file_name'] in subset_result:
            train_dataset.append(data)
        else:
            new_lake_data.append(data)

    #moving data from lake set to train set.
    change_dir(subset_result, src_dir, dest_dir)
    print('\n SHIFT TRAIN LEN :',len(train_dataset))

    #changing the file for annotations
    with open(lake_data_json, mode='w') as f:
        json.dump(new_lake_data,f,indent=4)
    with open(train_data_json,'w') as f:
        json.dump(train_dataset,f,indent=4)