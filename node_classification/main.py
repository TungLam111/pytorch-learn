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
            if calculate_cosine >= 0.99995:
                list_edges.append({"prob": calculate_cosine, "source": k ,"destination": j })
                list_src.append(k)
                list_des.append(j)
    print(data_matrix)
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
        self.conv3 = GraphConv(hidden_size, 9)
        self.conv4 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv3(g, h)
        h = torch.relu(h)
        h = self.conv4(g, h)
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
labeled_nodes = torch.tensor([0,15,20,25,40,45,60,    70,85,90,100,105,110,125,140])
labels = torch.tensor([0,0,0,0,0,0,0,   1,1,1,1,1,1,1,1])  # their labels are different

net = GCN(9, 9, nums_of_labels)
embed = nn.Embedding.from_pretrained(df_tensor)
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.005)
all_logits = []
for epoch in range(100):
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
draw(99)  # draw the prediction of the first epoch
plt.close()
"""

def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    return dgl.graph((u, v))

G = build_karate_club_graph()
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

nx_G = G.to_networkx().to_undirected()
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
G.ndata['feat'] = embed.weight

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

net = GCN(5, 5, 2)

inputs = embed.weight
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
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

"""