import torch
from os.path import join
from ai4code.data.dataset import load
from ai4code.data.dataloader import consecutive_cells

data_dir='/home/squirt/Documents/AI4Code'
#load(data_dir)

# Data loader
pk_file = join(data_dir, 'embeddings.pkl') 
ds = consecutive_cells(pk_file)
ds[0]
print('Done')

