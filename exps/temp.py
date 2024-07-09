import os 

# images = os.listdir('/data/circulars/DATA/layoutLM+Tactful/layoutlmv3_data/lake')

import json

data = json.load(open('/data/circulars/DATA/pix2struct_synth/final_dataset.json'))

print(len(data))
