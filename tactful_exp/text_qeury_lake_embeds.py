from encoder_2 import Pix2StructEncoder
import json
import os

with open('/data/circulars/DATA/pix2struct+tactful/data-1/docvqa_train.json') as f:
    query_data = json.load(f)


f_model = Pix2StructEncoder()
query_set_embeddings = f_model.get_embeds(
        query_data, '/data/circulars/DATA/pix2struct+tactful/data-1/train')


with open('/data/circulars/DATA/pix2struct+tactful/data-1/docvqa_val.json') as f:
    lake_data = json.load(f)
    
lake_set_embeddings = f_model.get_embeds(
        lake_data, '/data/circulars/DATA/pix2struct+tactful/data-1/val')