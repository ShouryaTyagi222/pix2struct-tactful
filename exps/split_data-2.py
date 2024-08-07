# python split_data-2.py -i '/data/circulars/DATA/pix2struct_synth/Dataset/jsons' -d '/data/circulars/DATA/pix2struct_synth/Dataset/Split_Images' -o '/data/circulars/DATA/pix2struct+tactful/data-1'

# Script to split the docvqa data set into trian, text and lake set
# this file only splits the data in the given file
# sample data in ./data-1

import random
import os
import json
import argparse
import warnings
import shutil
warnings.filterwarnings("ignore")
from PIL import Image
from transformers import AutoProcessor
print('RUNNING 1')


def main(args):
    data_folder=args.data_folder
    image_dir=arsgs.img_dir
    output_dir=args.output_dir
    train_split = int(args.train_length)
    val_split = int(args.val_length)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    for folder in ['train','lake','val']:
        if not os.path.exists(os.path.join(output_dir,folder)):
            os.mkdir(os.path.join(output_dir,folder))

    train_data=[]
    val_data=[]
    lake_data=[]
    skipped_images = []

    for file in os.listdir(data_folder):
        data_file_name = ''.join(file.split('.')[:-1])
        la=0
        va=0
        tr=0
        print('FILE NAME :', data_file_name)
        with open(os.path.join(data_folder, file), 'r') as f:
            data = json.load(f)

        for document in data:
            document = document[0]
            try:
                img_name=document['file_name']
                img_path=os.path.join(image_dir, data_file_name, img_name)
                img=Image.open(img_path)
                entries = []

                for q_a in document['question_answer_pairs']:
                    encoding = {}
                    encoding['answer'] = q_a['answer']
                    encoding['question'] = q_a['question']
                    encoding['file_name'] = img_name
                    # print(encoding['flattened_patches'])
                    entries.append(encoding)
                
                split=random.randint(0,2)

                if tr<train_split and split==0:
                    train_data.extend(entries)
                    tr+=1
                    img.save(os.path.join(output_dir,'train',img_name))
                elif va<val_split and split==1:
                    val_data.extend(entries)
                    va+=1
                    img.save(os.path.join(output_dir,'val',img_name))
                else:
                    lake_data.extend(entries)
                    la+=1
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

def parse_args():
    parser = argparse.ArgumentParser(description="Udop", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--data_folder", default=None, type=str, help="Path to the input json file")
    parser.add_argument("-d", "--img_dir", default=None, type=str, help="Path to the image Directory")
    parser.add_argument("-o", "--output_dir", default='/', type=str, help="Path to the Output Folder")
    parser.add_argument("-t", "--train_length", default='5', type=str, help="Train split of the data")
    parser.add_argument("-v", "--val_length", default='50', type=str, help="Val split of the data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    print('RUNNING')
    main(arg)