import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn.parameter import Parameter
from layers import GraphConvolution
from dgl.nn.pytorch.conv import GraphConv,GATConv,SAGEConv
import dgl

def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

class RNNGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RNNGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.Lambda = Parameter(torch.FloatTensor(1))
        self.Lambda.data.uniform_(0.2, 0.2)
        
        
    def forward(self, x, adj):
        #out=[]
        now_adj=adj[:,0,:].clone()
        for i in range(1,adj.shape[1]):  #time_steps
            now_adj=(1-self.Lambda)*now_adj+self.Lambda*adj[:,i,:]  #weight decay
        one_out=self.gc1(x[:,-1,:],now_adj)
        one_out=F.relu(one_out)

        one_out = F.dropout(one_out, self.dropout, training=self.training)
        one_out = self.gc2(one_out,now_adj)

        return F.log_softmax(one_out, dim=1)



class TRNNGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nnode,use_cuda=False):
        super(TRNNGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.Lambda = Parameter(torch.FloatTensor(nclass,nclass))
        self.Lambda.data.uniform_(0.5, 0.5)
        self.use_cuda=use_cuda
        
        y=torch.randint(0,nclass,(nnode,1)).flatten()
        
        if self.use_cuda:
            self.H = torch.zeros(nnode, nclass).cuda()
        else:
            self.H = torch.zeros(nnode, nclass)
        self.H[range(self.H.shape[0]), y]=1
  
        
    def forward(self, x, adj):

        w=self.Lambda.data
        w=w.clamp(0,1)
        self.Lambda.data=w
        if self.use_cuda:
            decay_adj=torch.mm(torch.mm(self.H,self.Lambda),self.H.T).cuda()
        else:
            decay_adj=torch.mm(torch.mm(self.H,self.Lambda),self.H.T)
        
        now_adj=adj[:,0,:].clone()#torch.zeros(adj.shape[0], adj.shape[2])
        for i in range(1,adj.shape[1]):  #time_steps
                now_adj=(1-decay_adj)*now_adj+decay_adj*adj[:,i,:]
        del decay_adj
        one_out=F.relu(self.gc1(x[:,-1,:],now_adj))

        one_out = F.dropout(one_out, self.dropout, training=self.training)
        one_out = self.gc2(one_out,now_adj)
        output=F.log_softmax(one_out, dim=1)
        y=torch.argmax(output,dim=1)
        H_shape=self.H.shape
        del self.H
        del now_adj
        if self.use_cuda:
            self.H = torch.zeros(H_shape).cuda()
        else:
            self.H = torch.zeros(H_shape)
        self.H[range(H_shape[0]), y]=1
        return output




class LSTMGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(LSTMGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.LS_begin=nn.LSTM(input_size=nfeat, hidden_size=nhid, num_layers=1, dropout=0.5,batch_first=True)

        self.nhid=nhid
        
        
    def forward(self, x, adj):
        adj=self.LS_begin(adj)
        x = F.relu(self.gc1(x[:,-1,:], adj[0][:,-1,:]))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj[0][:,-1,:])

        return F.log_softmax(x, dim=1)


class GCNLSTM(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNLSTM, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        
        self.LS_end=nn.LSTM(input_size=nhid, hidden_size=nclass, num_layers=2, dropout=0.5,
                            batch_first=True)
        self.nhid=nhid
        self.nclass=nclass
        self.linear=nn.Linear(nclass, nclass)
        
    def forward(self, x, adj):
        out=[]
        for i in range(adj.shape[1]):
            one_out=F.relu(self.gc1(x[:,i,:],adj[:,i,:]))
            one_out = F.dropout(one_out, self.dropout, training=self.training)
            one_out = self.gc2(one_out, adj[:,i,:])
            out+=[one_out]
        out = torch.stack(out, 1)   
        out=self.LS_end(out)[0][:,-1,:]

        return F.log_softmax(out, dim=1)










class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
        
    def forward(self, x, adj):
        

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        

        return F.log_softmax(x, dim=1)


    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(nfeat, nhid, num_heads=1)
        self.conv2 = GATConv(nhid, nclass, num_heads=1)
        

    def forward(self, x, adj):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # Perform graph convolution and activation function.
        
        x = F.relu(self.conv1(adj, x))  #different from self-defined gcn
        x=x.reshape(x.shape[0],x.shape[2])
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(adj, x)
        x=x.reshape(x.shape[0],x.shape[2])
        
        return F.log_softmax(x, dim=1)

    
class GraphSage(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(nfeat, nhid,aggregator_type='mean')
        self.conv2 = SAGEConv(nhid, nclass,aggregator_type='mean')
        

    def forward(self, x, adj):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # Perform graph convolution and activation function.
        x = F.relu(self.conv1(adj, x))  #different from self-defined gcn
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(adj, x)
        
        return F.log_softmax(x, dim=1)
    
    
    
    
    
#egcn
    
class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)
        
def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect




#only use EGCN

class EGCN(torch.nn.Module): #egcn_o
    def __init__(self, nfeat, nhid, nclass, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = Namespace({})

        feats = [nfeat,
                 nhid,
                 nhid]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = nhid,out_features = nhid),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(in_features = nhid,out_features = nclass))
        for i in range(1,len(feats)):
            GRCU_args = Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': torch.nn.RReLU()})

            grcu_i = GRCU(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))
        
    def parameters(self):
        return self._parameters

    def forward(self,Nodes_list, A_list):#,nodes_mask_list):
        node_feats= Nodes_list[-1]
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list)#,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
       
        
        return F.log_softmax(self.mlp(out), dim=1)


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)  

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list):#,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights)#,node_embs,mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q):#,prev_Z,mask):     ###Same as GCNH
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q
        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap
        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()


