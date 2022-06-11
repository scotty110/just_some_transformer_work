'''
Make a data loader for training:
    - Will have 2 examples, swap dataset and consecutive dataset
'''

import numpy as np
import torch
import pickle

from torch.utils.data import Dataset

class consecutive_cells(Dataset):
    '''
    This dataset is going to be slightly different in it will generate "new"
    data after each epoch. 
    '''
    def __init__( self,
                    file:str,
                    num_pos:int=1,
                    num_neg:int=3,
                    f16:bool=False):
        # Load Data
        self.file_name = file
        self.data = self.load_data()
    
        # Generate training pairs
        self.num_pos = num_pos
        self.num_neg = num_neg
        if self.num_neg < 1 or self.num_pos < 1:
            raise Exception("num pos/neg needs to be greater than 2 to form a +- pair")
        self.training_data = self.select_data()

        # Are we training w/ float16 (in on cuda, yes)
        self.f16 = f16

    def load_data(self)->dict:
        # Just load Pickel file
        with open(self.file_name, 'rb') as handle:
            data = pickle.load(handle) 
        return data

    def select_data(self)->list:
        pair_list = [ None for i in range( len(self.data)*(self.num_pos + self.num_neg))]
        pair_count = 0 # counter for pair_list
        for _,arr in self.data.items():
            if (arr.shape[1]!=768):
                print( arr.shape )
                raise ValueError("Something went wrong")

            # Generate Positive examples
            for i in range(self.num_pos):
                r = np.random.randint(0,arr.shape[0])
                p1 = arr[r,:]
                p1 = np.reshape(p1, (1,*p1.shape))
                if r == arr.shape[0]-1:
                    p2 = np.zeros(p1.shape) #p1 was already reshaped!
                else:
                    p2 = arr[r+1,:]
                    p2 = np.reshape(p2, (1,*p2.shape))
                X = np.append(p1, p2, axis=0)
                Y = 1.
                pair_list[pair_count] = (X,Y)
                pair_count += 1

            # Generate Negative Examples
            for i in range(self.num_neg):
                # Generate Random pairs
                r1 = np.random.randint(0,arr.shape[0])  
                r2 = np.random.randint(0,arr.shape[0])  
                while r1 == r2:
                    r2 = np.random.randint(0,arr.shape[0])  
                # Pairs
                p1 = arr[r1,:]
                p1 = np.reshape(p1, (1,*p1.shape))
                p2 = arr[r2,:]
                p2 = np.reshape(p2, (1,*p2.shape))
                X = np.append(p1, p2, axis=0)
                Y = 0.
                pair_list[pair_count] = (X,Y)
                pair_count += 1
        return pair_list


    def __len__(self):
        return len(self.training_data)

    def __getitem__(self,index)->tuple:
        '''
        Preforms the actual get data
        Inputs:
            - index (int): The index in self.training_data to return
        Returns:
            A tuple in the format (X,y) where X is a 2xSize tensor, y=0/1
        '''
        nX,nY = self.training_data[index]
        X = torch.tensor(nX)
        y = torch.tensor(nY)

        if self.f16:
            X = X.half()
            y = y.half()
        return (X,y)







