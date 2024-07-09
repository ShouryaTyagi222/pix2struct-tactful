import os, shutil, json
import numpy as np
import sys
sys.path.append("../")
import sys
sys.path.append("../")
    

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
            # os.mkdir(dest_dir[1])

        try:
            if image not in os.listdir(dest_dir[0]):
                shutil.copy(source_img, destination_img)
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
            return

        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
            return

        # For other errors
        except Exception as e:
            print("Error occurred while copying file.", e)
            return


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