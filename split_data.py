# file to split the data fraom a the file /data/circulars/DATA/pix2struct_synth/final_dataset.json


import os
import json
import shutil
from PIL import Image
import random

input_file = '/data/circulars/DATA/pix2struct_synth/final_dataset.json'
output_dir = '/data/circulars/DATA/pix2struct+tactful/data-1'

train_size = 20
val_size = 200

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
else:
    os.mkdir(output_dir)


for folder in ['train','lake','val']:
    if not os.path.exists(os.path.join(output_dir,folder)):
        os.mkdir(os.path.join(output_dir,folder))

train_data=[]
val_data=[]
lake_data=[]
skipped_images = []

train_images = []
val_images = []
lake_images = []

la=0
va=0
tr=0
with open(os.path.join(input_file), 'r') as f:
    data = json.load(f)

random.shuffle(data)

for item in data:
    try:
        img_path=item['document']
        img_name=os.path.basename(img_path)
        encoding = {}
        encoding['answer'] = item['answer']
        encoding['question'] = item['question']
        encoding['file_name'] = img_name

        if img_name in train_images:
            train_data.append(encoding)
        elif img_name in val_images:
            val_data.append(encoding)
        elif img_name in lake_images:
            lake_data.append(encoding)

        elif tr!=train_size:
            train_images.append(img_name)
            train_data.append(encoding)
            shutil.copy(img_path, os.path.join(output_dir,'train',img_name))
            tr+=1
        elif va!=val_size:
            val_images.append(img_name)
            val_data.append(encoding)
            shutil.copy(img_path, os.path.join(output_dir,'val',img_name))
            va+=1
        else:
            lake_images.append(img_name)
            lake_data.append(encoding)
            shutil.copy(img_path, os.path.join(output_dir,'lake',img_name))
            la+=1
        
        print(img_name)
        print([tr,va,la])
        print([len(train_data),len(val_data),len(lake_data)])
        
    except Exception as e:
        print(e)
    
print('IMAGES SKIPPED :',skipped_images)
print('SKIPPED IMAGES LENGTH :',len(skipped_images))

with open(os.path.join(output_dir,'docvqa_train.json'), 'w') as json_file:
    json.dump(train_data, json_file,indent=4)
with open(os.path.join(output_dir,'docvqa_val.json'), 'w') as json_file:
    json.dump(val_data, json_file,indent=4)
with open(os.path.join(output_dir,'docvqa_lake.json'), 'w') as json_file:
    json.dump(lake_data, json_file,indent=4)


