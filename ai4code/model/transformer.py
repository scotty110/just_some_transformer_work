'''
Make transformer parts
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class scaled_dot(nn.Module):
    def forward(self, Q, K, V):
        # Q,K,V are all matricies (512x512)
        attention = torch.matmul( 
                        F.softmax(
                            torch.matmul(Q, K.transpose(0,1)) / K.shape[0],
                            dim=0
                        ),
                        V
                    )
        return attention


class single_multi_head(nn.Module):
    def __init__(self, q_shape, k_shape, v_shape):
        super().__init__()
        # k,q,v are all vectors???
        self.q_linear = nn.Linear(q_shape[1], q_shape[1])
        self.k_linear = nn.Linear(k_shape[1], k_shape[1])
        self.v_linear = nn.Linear(v_shape[1], v_shape[1])
        self.s_dot = scaled_dot()

    def forward(self, q, k, v):
        attention = self.s_dot(
                        self.q_linear(q),
                        self.k_linear(k),
                        self.v_linear(v),
                    )
        return attention


class multi_head(nn.Module):
    def __init__(self, q_shape, k_shape, v_shape, heads:int=3):
        super().__init__()
        self.heads = nn.ModuleList(
            [ single_multi_head(q_shape, k_shape, v_shape) for i in range(heads)]       
        )
        self.w = nn.Linear((heads*v_shape[1]), v_shape[1])

    def forward(self, q, k, v):
        heads = torch.concat( [self.heads[i](q,k,v) for i in range(len(self.heads))], dim=1)
        m_attention = self.w(heads)
        return m_attention

class decode_layer(nn.Module):
    def __init__(self, vector_shape):
        super().__init__()
        # Block 1
        self.mh_1 = multi_head(vector_shape, vector_shape, vector_shape)
        self.ln_1 = nn.LayerNorm( vector_shape )
       
        # Block 2
        self.mh_2 = multi_head(vector_shape, vector_shape, vector_shape)
        self.ln_2 = nn.LayerNorm( vector_shape )

        # Block 3
        self.w1 = nn.Linear(vector_shape[1], vector_shape[1])
        self.ln_3 = nn.LayerNorm( vector_shape )

    def forward(self, in_encoding, out_encoding):
        block_1 = self.ln_1(
                    self.mh_1(out_encoding, out_encoding, out_encoding) + out_encoding
                  )  

        block_2 = self.ln_2(
                    self.mh_2(in_encoding, in_encoding, block_1) + out_encoding
                  )  

        block_3 = self.ln_3( self.w1( block_2 ) + block_2 )
        return block_3


class decoder(nn.Module):
    def __init__(self, vector_shape, probs):
        super().__init__()
        # Both input and output vectors are the same shape
        self.d_layers = nn.ModuleList(
            [ decode_layer(vector_shape) for i in range(6)]
        )

        self.w = nn.Linear(vector_shape[1], probs)

    def forward(self, enc_v, out_v):
        x = out_v
        for f in self.d_layers:
            x = f( enc_v, x)
        #return F.softmax( self.w(x), dim=1 )
        return self.w(x)

        
if __name__ == '__main__':
    device = torch.device('cuda:0')
    k = torch.rand(1,728).to(device, torch.half)
    o = torch.ones(1,728).to(device, torch.half)
    p = torch.tensor([.5]).to(device, torch.half)

    d_coder = decoder( k.shape, 1).to(device, torch.half)
    loss_func = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(
                    d_coder.parameters(),
                    0.0005
                )

    for i in range(100):
        output = d_coder(k, o)
        loss = loss_func(output, p)
        print(loss.item())

        # "Learn"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final Loss 
    d_coder.eval()
    loss = loss_func(d_coder(k), p)
    print('Final Loss: {}'.format(loss.item()))
    print('done')
        





