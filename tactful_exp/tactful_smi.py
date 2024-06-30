from .strategy import Strategy
import numpy as np

from .encoder_3 import Pix2StructEncoder

import torch
from torch import nn
from scipy import stats
import submodlib
import os
import json

class TACTFUL_SMI(Strategy):
    def __init__(self, labeled_dataset=None, unlabeled_dataset=None, net=None, f_net=None, nclasses=None, args={}): #
        
        super(TACTFUL_SMI, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        

    def select(self, budget, query_file, lake_file, query_images, lake_images, processor):

        model_path = self.args['model_path']
        clazz = self.args['class']
        iteration = self.args['iteration']
        eta = self.args['eta']
        optimizer = self.args['optimizer'] if 'optimizer' in self.args else 'NaiveGreedy'
        metric = self.args['metric'] if 'metric' in self.args else 'cosine'
        eta = self.args['eta'] if 'eta' in self.args else 1
        stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        verbose = self.args['verbose'] if 'verbose' in self.args else False
        print('runnign s : ')
        
        with open(query_file) as f:
            query_data = json.load(f)

        f_model = Pix2StructEncoder(processor)
        query_set_embeddings = f_model.get_embeds(
                query_data, query_images)

        with open(lake_file) as f:
            lake_data = json.load(f)
            
        lake_set_embeddings = f_model.get_embeds(
                lake_data, lake_images)
        

        query_embedding = []

        for idx, query in enumerate(query_set_embeddings, start=1):
            q_img, q_hist = query['img'], query['hist']
            query_embedding.append(q_hist)

        lake_embedding = []
        lake_image_list = []
        for idx, sample in enumerate(lake_set_embeddings, start=1):
            s_img, s_hist = sample['img'], sample['hist']
            lake_embedding.append(s_hist)
            lake_image_list.append(s_img)

        # print('LAKE EMBEDDINGS :', lake_embedding)
        # print('QUERY EMBEDDINGS :', query_embedding)
        

        if (len(lake_embedding) < budget):
            budget = len(lake_embedding) - 1

        lake_embedding = torch.tensor(lake_embedding)
        query_embedding = torch.tensor(query_embedding)
        if(self.args['smi_function']=='fl1mi'):
            obj = submodlib.FacilityLocationMutualInformationFunction(n=lake_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=lake_embedding , 
                                                                      queryData=query_embedding, 
                                                                      magnificationEta=eta)

        if(self.args['smi_function']=='fl2mi'):
            print('Runnign fl2mi')
            obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=lake_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=lake_embedding,
                                                                      queryData=query_embedding,
                                                                      queryDiversityEta=eta)
        
        if(self.args['smi_function']=='com'):
            from submodlib_cpp import ConcaveOverModular
            obj = submodlib.ConcaveOverModularFunction(n=lake_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=lake_embedding,
                                                                      queryData=query_embedding, 
                                                                      queryDiversityEta=eta,
                                                                      mode=ConcaveOverModular.logarithmic)
        if(self.args['smi_function']=='gcmi'):
            obj = submodlib.GraphCutMutualInformationFunction(n=lake_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0],
                                                                      data=lake_embedding,
                                                                      queryData=query_embedding, 
                                                                      metric=metric)
        if(self.args['smi_function']=='logdetmi'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.LogDeterminantMutualInformationFunction(n=lake_embedding.shape[0],
                                                                    num_queries=query_embedding.shape[0],
                                                                    data=lake_embedding,  
                                                                    queryData=query_embedding,                                                                    
                                                                    magnificationEta=eta,
                                                                    lambdaVal=lambdaVal)
        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]
        return lake_image_list, greedyIndices

if __name__ == "__main__":
    strategy_sel = TACTFUL_SMI(args = {"class":None, 'eta':1, "model_path":'/', 'smi_function':'fl2mi', 'iteration' : 1})
    lake_image_list, subset_result = strategy_sel.select(10)
    print(lake_image_list)