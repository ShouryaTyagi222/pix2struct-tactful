# Pix2Struct+Tactful

## config
`configs.py` consists of the required configs for the model training

## train
```
python train.py
```
## infer
```
python infer.py -i <INPUT_IMG_PATH> 
```

Note :
- the data is in data-1 folder
- to prepare data for training run preprocess.py
- to split the data in train, lake, val run split_data_3.py
- The outputs(logs and model) are saved in model_outputs/output_2 folder.


## Setup

- conda install pytorch torchvision torchaudio cudatoolkit=11.5 -c pytorch