'''
Make a dataset for pytorch

Info on dataset: 
    https://www.kaggle.com/code/erikbruin/ai4code-find-your-stuff-kendall-tau-and-eda/data
'''
import pandas as pd
import json
from os.path import join
import glob


def load_csv(file_path:str)->pd.DataFrame:
    '''
    Load csv file with ID, order ...
    '''
    df = pd.read_csv(file_path)
    #print(df.iloc[0])
    rd = df.loc[0].to_dict()

    df['cell_order'] = df['cell_order'].apply( lambda x: x.split(' '), )
    rd = df.loc[0].to_dict()
    print(rd)
    
    return

def load_json(dir_path:str)->list:
    files = glob.glob( join(data_dir,json_dir,'**/*.json'), recursive=True ) 
    return files

def load(data_dir:str):
    '''
    Load all
    '''
    order_file = 'train_orders.csv'
    json_dir = 'train'
    _ = load_csv( join(data_dir, order_file) )
    return
