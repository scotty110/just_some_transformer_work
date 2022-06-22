import torch
from os.path import join
#from ai4code.data.dataset import load
#from ai4code.data.dataloader import consecutive_cells
from ai4code.data.dataloader import get_dataloaders 

#from ai4code.model import transformer

if __name__ == '__main__':
    data_dir='/home/squirt/Documents/AI4Code'
    #load(data_dir)

    # Data loader
    # Memory issues, do lazy loading
    emb_file = join(data_dir, 'embeddings_fp16.pkl') 
    key_file = join(data_dir, 'embeddings_keys.pkl')
   
    t_dl, e_dl = get_dataloaders(emb_file, key_file)
    print( len(t_dl) )
    print( len(e_dl) )

    
    # Test transformer
    #_ = transformer.test()

    print('Done')

