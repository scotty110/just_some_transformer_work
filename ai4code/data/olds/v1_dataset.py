'''
Make a dataset for pytorch

Info on dataset: 
    https://www.kaggle.com/code/erikbruin/ai4code-find-your-stuff-kendall-tau-and-eda/data
'''
import csv
import json
from os.path import join
import glob

# Might as well
#from multiprocessing import Pool

def split_csv_row(row:list)->dict:
    '''
    Turn [ID, cell_order ,...]
    Into {json_id:[cell_order, ...]}
    '''
    r=row.split(' ')
    d = {r[0]:r[1:]} 
    return d

def load_csv(file_path:str)->dict:
    '''
    Load csv file with ID, order ...
    '''
    
    with open( file_path, newline='\n' ) as f:
        reader = csv.reader(f)
        data = list(reader)
    print( data[0] )
    print( type(data[0]))
    map_results = map(split_csv_row, data) 
    print(list(map_results))

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
