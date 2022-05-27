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
    df['cell_order'] = df['cell_order'].apply( lambda x: x.split(' '), )
    rd = df.loc[0].to_dict()
    print(rd)
    
    return

def get_json(dir_path:str)->list:
    ''' Find all json files '''
    files = glob.glob( join(dir_path,'**/*.json'), recursive=True ) 
    return files

def load_json(file_path:str, only_code=True):
    with open(file_path) as json_file:
        data = json.load(json_file)

    # If we only want code (this might be nice since code is a graph)
    if only_code:
        cell_data = dict(filter(lambda x: x[1] == 'code', data['cell_type'].items()))
    else:
        cell_data = data['cell_type']

    k = set(cell_data.keys())
    # Filter source for what we want (if using only code)
    source = dict(filter(lambda x: x[0] in k, data['source'].items()))
    print(source.keys())
    '''
    for i in data['cell_type']:
        print(type(i))
        print(i)
    '''
    return
    

def load(data_dir:str):
    '''
    Load all
    '''
    order_file = 'train_orders.csv'
    json_dir = 'train'
    #_ = load_csv( join(data_dir, order_file) )
    json_files = get_json( join(data_dir, json_dir) )
    print(json_files[0])
    load_json(json_files[0])
    return
