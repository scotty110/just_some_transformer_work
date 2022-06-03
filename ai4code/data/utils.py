'''
Tools for datasets

Containes:
    - embedding support
'''

import re
import numpy as np
from sentence_transformers import SentenceTransformer

class transformer():
    ''' Turn A Sentence into an Embedding '''
    def __init__(self, model_name='all-mpnet-base-v2', length=384):
        ''' See: https://www.sbert.net/docs/pretrained_models.html '''
        self.model = SentenceTransformer(model_name)
        self.s_len = length

    def pre_process(self, s_list):
        to_return = [None for i in range(len(s_list))]
        for i,s in enumerate(s_list): 
            new_string = s.replace('\n',' ')
            new_list = re.split(r' ', new_string)
            if len(new_list)>self.s_len:
                a = ' '.join( new_list[:int(self.s_len/2)] )
                b = ' '.join( new_list[-int(self.s_len/2):] )
                new_list = a+b 
            to_return[i] = ' '.join(new_list)
        return to_return

    def embedding(self, s):
        s = self.pre_process(s)
        embedding = np.array( self.model.encode(s) )
        return embedding

'''
t = transformer()
a = t.embedding(['Hello\nWorld Best Day','hello world'])
print(a)
'''
