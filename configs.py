import os
import torch

# Basic args for Tactful 
args = {
    "strategy":'fl2mi',      # strategy to be used for tactful
    "total_budget":1395,  # Total data points available to transfer from lake to train set
    # "lake_size":1395,  # Size of the lake dataset
    # "train_size":20,  # Size of the training dataset
    "device":0,   # GPU Device
    "proposal_budget":10,  # Budget for Selection of Lake images
    "iterations":139,   # Total AL Rounds
    "batch_size":8    # Batch Size
}


# Name of the model to be trained layoutlmv2/layoutlmv3
model_name = 'pix2struct'    # Name of the Model
learning_rate = 1e-5         # Learning Rate of the Model Training
epoch_t=1                    # The Epoch count after which the the model starts saving the sample question and answers for every 5 epochs
init_epochs = 30                # The Epoch count of the Initial Training
MAX_AL_EPOCHS = 10               # The Epoch count of Each Active Learning round
ROUGE_THRESH = 0.95          # Threshold for Training a Round BAsed on Rouge-L F1 Score
TRAIN_LOSS_THRESH = 0.3      # Threshold for Training a Round BAsed on Train Loss
EPOCH_THRESH = 1            # Threshold for Minimum Epoch in a Round
SCHEDULER_T_MAX = 30        # Steps for the Scheduler
SATURATION_THRESH = 5       # Minimum epochs for Saturated Train Loss

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


# Requried data paths
train_path = '/data/circulars/DATA/pix2struct+tactful/model_outputs/output_2'    # path of the output dir
data_dir = '/data/circulars/DATA/pix2struct+tactful/data-1' # path to the splitted data
# query_path = '/data/circulars/DATA/layoutLM+Tactful/query_images'  # path to the query Images Folder

INIT_MODEL_PATH = 'google/pix2struct-docvqa-base'   # Initial model Path
PROCESSOR_PATH = "ybelkada/pix2struct-base"

# full_data_annots = '/data/circulars/DATA/pix2struct+tactful/full_data_annots.json'  # Path to the file consisting of the class wise annotations of the data
IMAGE_DIR = '/data/circulars/DATA/LayoutLM/docvqa_dataset/Images'   # Path to the Image Directory

# Wandb Credentials
wandb_flag=1
wandb_project_desc=f'pix2struct_Tactful_FineTuning_{model_name}'
wandb_name=f'{os.path.basename(train_path)}_{args["strategy"]}_{learning_rate}_{args["batch_size"]}_{ROUGE_THRESH}'
wandb_model_desc=f'{model_name} - Fine Tuned on DocVQA using pix2struct+tactful'
wandb_key='ead46cf543385050fcec224a0c2850faffcae584'

train_data_dirs = (os.path.join(data_dir,"train"),
                   os.path.join(data_dir,"docvqa_train.json"))
lake_data_dirs = (os.path.join(data_dir,"lake"),
                  os.path.join(data_dir,"docvqa_lake.json"))
val_data_dirs = (os.path.join(data_dir,"val"),
                 os.path.join(data_dir,"docvqa_val.json"))

# query_path = os.path.join(query_path, args['category'])

args["output_path"] = args['strategy']
training_name = args['output_path']
model_path = os.path.join(train_path, training_name)
def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass
if (not os.path.exists(model_path)):
    create_dir(model_path)
output_dir = os.path.join(model_path, "initial_training")

selection_arg = {'eta':1, "model_path":model_path, 'smi_function':args['strategy']}

if torch.cuda.is_available():
    torch.cuda.set_device(args['device'])