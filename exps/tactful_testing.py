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

iteration = args['iterations']
selection_strag = args['strategy']
selection_budget = args['budget']
budget = args['total_budget']
proposal_budget = args['proposal_budget']

i=0
processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)   

if (selection_strag != "random"):
    print('DOING TACTFUL')
    selection_arg['iteration'] = i
    strategy_sel = TACTFUL_SMI(args = selection_arg)
    lake_image_list, subset_result = strategy_sel.select(proposal_budget, train_data_dirs[1], val_data_dirs[1], train_data_dirs[0], val_data_dirs[0], processor)
    print('LENGTH OF SUBSET RESULT :',subset_result)
    subset_result = [lake_image_list[i] for i in subset_result]
    subset_result = list(
        set(subset_result))
    print(subset_result)

# subset_result = ['img_fp_Synthetic_19316.png', 'img_mp_196_saralshabdavali_Civil Aviation.png', 'img_mp_498_saralshabdavali_Civil Aviation.png', 'img_fp_Synthetic_HRM2_Servicematters_16126_Hindi.png', 'img_lp_Ministry of Education_Internship_2015.png', 'img_fp_Synthetic_PFD_RenewalLease_3187.png', 'img_mp_46_saralshabdavali_Civil Aviation.png', 'img_fp_Rajabhasha Department_ctb6dec18eng_0.png', 'img_lp_Kerela Higher Ed_103-17%20%20Circular.png', 'img_fp_Department of Expenditure_09-01-1976.png', 'img_fp_Synthetic_926.png', 'img_lp_Department of Personal Training_levelD17092018to07122018wZ3b2.png', 'img_mp_60_GuidelineAct2002_Co-operation Department Multi State.png', 'img_fp_Synthetic_4749.png', 'img_lp_Civil Aviation_moca_003080_0.png', 'img_lp_Ministry of Road Transport and Highways_Notification_no_GSR_178E_dated_20_2_2018_regarding_supplementary_notification_to_BS__VI.png']
if (budget > 0):
    aug_train_subset(subset_result, train_data_dirs[1], val_data_dirs[1], budget, val_data_dirs, train_data_dirs)
    with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
        f.write(f'LENGTH OF THE TRAIN DATA : {len(os.listdir(train_data_dirs[0]))}\n')
    print('LENGTH OF THE TRAIN DATA : ',len(os.listdir(train_data_dirs[0])))
    
i+=1