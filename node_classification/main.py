import dgl
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import random 
from scipy import spatial
from ogb.nodeproppred import NodePropPredDataset, DglNodePropPredDataset
import networkx as nx
from dgl.nn.pytorch import GraphConv
import itertools
import matplotlib.animation as animation
import matplotlib.pyplot as plt

nums_of_samples = 146
nums_of_labels = 2
feature_size = 9
labels = [0,1]

def process_data():
    df_train = pd.read_csv("glass_data.csv")
    df_node = df_train["column_a"]
    df_x = df_train.drop(["column_k", "column_a"], axis = 1) 
    df_y = df_train["column_k"]
    
    data = []
    data_matrix = np.zeros(shape=(nums_of_samples, nums_of_samples))
    list_edges = [] # ({"prob": prob ,"source": src, "destination": des})
    list_src = []
    list_des = []

    for i in range(nums_of_samples):
        vector_row = df_x.iloc[i] 
        data.append(vector_row)
    df_tensor = torch.DoubleTensor(data)
  
    for k in range(nums_of_samples):
        for j in range(k, nums_of_samples):
            calculate_cosine = cosine_similarity2(df_tensor[k], df_tensor[j])
            data_matrix[k][j] = calculate_cosine
            if calculate_cosine >= 0.9997 :
                list_edges.append({"prob": calculate_cosine, "source": k ,"destination": j })
                list_src.append(k)
                list_des.append(j)

    return df_tensor, df_node - 1, df_x, df_y, list_edges , list_src, list_des

def cosine_similarity2(target_vec, vec):
    return 1 - spatial.distance.cosine(target_vec, vec)

def build_graph(list_src, list_des):
    src = np.array(list_src)
    dst = np.array(list_des)

    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])

    return dgl.graph((u, v))

def data_loader_dgl_node_ogb(d_name):
    dataset = DglNodePropPredDataset(name = d_name)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0]
    return graph, label

def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    results = []
    for v in range(nums_of_samples):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        results.append(cls)
        colors.append(cls1color if cls else cls2color)
    print(results)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv3(g, h)
        return h

df_tensor, df_node, df_x, df_y, list_edges , list_src, list_des = process_data()
G = build_graph(list_src, list_des)
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())
G.ndata['feat'] = df_tensor
    
nx_G = G.to_networkx().to_undirected()

pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

df_tensor = df_tensor.to(dtype=torch.float32)

inputs = df_tensor
labeled_nodes = torch.tensor([0,15,20,25,40,45,60,    70,85,90,105,110,125,140])
labels = torch.tensor([0,0,0,0,0,0,0,   1,1,1,1,1,1,1])  # their labels are different

net = GCN(9, 16, nums_of_labels)
embed = nn.Embedding.from_pretrained(df_tensor)
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.001)
all_logits = []
for epoch in range(50):
    logits = net(G, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    
fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
draw(49)  # draw the prediction of the first epoch
plt.close()



























"""
1
1
0
1
0
0
0
1
0
0
1
1
1
1
0
0
1
1
0
1
0
1
0
0
0
1
0
1
0
0
0
0
0
0
"""