# file to split the data fraom a the file /data/circulars/DATA/pix2struct_synth/final_dataset.json


import os
import json
import shutil
from PIL import Image
import random

input_file = '/data/circulars/DATA/pix2struct_synth/final_dataset.json'
output_dir = '/data/circulars/DATA/pix2struct+tactful/data-2'

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

la=0
va=0
tr=0
with open(os.path.join(input_file), 'r') as f:
    data = json.load(f)

for item in data:
    try:
        img_path=item['document']
        img=Image.open(img_path)
        img_name=os.path.basename(img_path)
        encoding = {}
        encoding['answer'] = item['answer']
        encoding['question'] = item['question']
        encoding['file_name'] = img_name
        # print(encoding['flattened_patches'])
        # entries.append(encoding)
        
        split=random.randint(0,2)

        if tr<407 and split==0:
            train_data.append(encoding)
            tr+=1
            if img_name not in os.listdir(os.path.join(output_dir,'train')):
                img.save(os.path.join(output_dir,'train',img_name))
        elif va<8064 and split==1:
            val_data.append(encoding)
            va+=1
            if img_name not in os.listdir(os.path.join(output_dir,'val')):
                img.save(os.path.join(output_dir,'val',img_name))
        else:
            lake_data.append(encoding)
            la+=1
            if img_name not in os.listdir(os.path.join(output_dir,'lake')):
                img.save(os.path.join(output_dir,'lake',img_name))

        print(img_name)
        print([tr,va,la])
        print([len(os.listdir(output_dir+'/train')),len(os.listdir(output_dir+'/val')),len(os.listdir(output_dir+'/lake'))])
    except Exception as e:
        skipped_images.append(img_name)
        print(e)
        print('IMAGES SKIPPED :',skipped_images)
        print('SKIPPED IMAGES LENGTH :',len(skipped_images))

        with open(os.path.join(output_dir,'docvqa_train.json'), 'w') as json_file:
            json.dump(train_data, json_file,indent=4)
        with open(os.path.join(output_dir,'docvqa_val.json'), 'w') as json_file:
            json.dump(val_data, json_file,indent=4)
        with open(os.path.join(output_dir,'docvqa_lake.json'), 'w') as json_file:
            json.dump(lake_data, json_file,indent=4)
        break


print('IMAGES SKIPPED :',skipped_images)
print('SKIPPED IMAGES LENGTH :',len(skipped_images))

with open(os.path.join(output_dir,'docvqa_train.json'), 'w') as json_file:
    json.dump(train_data, json_file,indent=4)
with open(os.path.join(output_dir,'docvqa_val.json'), 'w') as json_file:
    json.dump(val_data, json_file,indent=4)
with open(os.path.join(output_dir,'docvqa_lake.json'), 'w') as json_file:
    json.dump(lake_data, json_file,indent=4)
