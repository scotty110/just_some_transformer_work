'''
Make a dataset for pytorch

Info on dataset: 
    https://www.kaggle.com/code/erikbruin/ai4code-find-your-stuff-kendall-tau-and-eda/data
'''
#import pandas as pd
import csv
import json
import pickle
from os.path import join
import glob
from ai4code.data.utils import transformer


def split_csv_row(row:list)->dict:                                              
    '''                                                                         
    Turn [ID, cell_order ,...]                                                  
    Into {json_id:[cell_order, ...]}                                            
    '''                                                                         
    r=row[1].split(' ')                                                            
    d = {row[0]:r}                                                            
    return d                                                                    

def load_csv(file_path:str)->dict:                                              
    '''                                                                         
    Load csv file with ID, order ...                                            
    '''                                                                         
    with open( file_path, newline='\n' ) as f:                                  
        reader = csv.reader(f)                                                  
        data = list(reader)                                                     
    # First Element is telling us (ID, ORDER)
    data.pop(0)

    results = dict((key, val) for k in list(map(split_csv_row, data)) for key, val in k.items())
    
    #map_results = list(map(split_csv_row, data))} 
    return results 


def get_json(dir_path:str)->list:
    ''' Find all json files '''
    files = glob.glob( join(dir_path,'**/*.json'), recursive=True ) 
    file_dict = { f.split('/')[-1].split('.')[0]:f for f in files }
    return file_dict

def load_json(file_path:str, only_code=True):
    '''
    Load json file. Turn into dict, with cell_id:string_value
    '''
    with open(file_path) as json_file:
        data = json.load(json_file)

    # If we only want code (this might be nice since code is a graph)
    # Maybe we add the identifiers back in later???
    if only_code:
        cell_data = dict(filter(lambda x: x[1] == 'code', data['cell_type'].items()))
        k = set(cell_data)
        source = dict(filter(lambda x: x[0] in k, data['source'].items()))
    else:
        source = data['source']
    return source


def convert_json(ID, json_file, orders, tform)->dict:
    ''' Load JSON, and return a dict by cell ID with embedding and place as tuple'''
    #ID = json_file.split('/')[-1].split('.')[0]
    string_keys = load_json(json_file)

    cell_order = orders[ID]
    embedding_dict = {}
    i=0
    to_process = [None for i in range(len(cell_order)) ]
    for n in cell_order: 
        if n in string_keys.keys():
            i+=1
            to_process[i] = string_keys[n]

    # Filter out None
    to_process = list(filter(lambda x: x!=None, to_process))

    # Generate Embeddings (this might be faster to batch lists)
    embeddings = tform.embedding( to_process )
    
    # Turn Embeddings back into dick with number values
    #embedding_dict[n] = { (i,emb) for i,emb in enumerate(embeddings) }
    #return ID, embedding_dict 
    return ID, embeddings 

def load(data_dir:str):
    '''
    Load all
    '''
    order_file = 'train_orders.csv'
    json_dir = 'train'
    id_order = load_csv( join(data_dir, order_file) )
    json_files = get_json( join(data_dir, json_dir) )
    tform = transformer()

    '''
    jfile = '/home/squirt/Documents/AI4Code/train/6323080c5d2b46.json'
    json_file = convert_json(jfile, id_order, tform)
    print(json_file)
    '''
    dataset = {}
    for file_id, file_name in json_files.items():
        k,v = convert_json(file_id, file_name,id_order,tform)
        dataset[k] = v 

    # Save Results (apperently compute take some time)
    with open( join(data_dir,'embeddings.pkl'), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return
