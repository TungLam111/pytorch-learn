import dgl.data
from dgl.nn import GraphConv
import dgl
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
dataset = dgl.data.CoraGraphDataset()
print('Number of categories:', dataset.num_classes)

g = dataset[0]

print('Node features')
print(g.ndata)
print('Edge features')
print(g.edata)

class pca:
    def __init__(self, n_components):
        """
        :param n_components: Number of principal components the data should be reduced too.
        """
        self.components = n_components

    def fit_transform(self, X):
        """
        * Centering our inputs with mean
        * Finding covariance matrix using centered tensor
        * Finding eigen value and eigen vector using torch.eig()
        * Sorting eigen values in descending order and finding index of high eigen values
        * Using sorted index, get the eigen vectors
        * Tranforming the Input vectors with n columns into PCA components with reduced dimension
        :param X: Input tensor with n columns.
        :return: Output tensor with reduced principal components
        """
        centering_X = X - torch.mean(X, dim=0)
        covariance_matrix = torch.mm(centering_X.T, centering_X)/(centering_X.shape[0] - 1)
        eigen_values, eigen_vectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigen_sorted_index = torch.argsort(eigen_values[:,0],descending=True)
        eigen_vectors_sorted = eigen_vectors[:,eigen_sorted_index]
        component_vector = eigen_vectors_sorted[:,0:self.components]
        transformed = torch.mm(component_vector.T, centering_X.T).T
        return transformed
pca = pca(n_components=3)
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
     #   self.conv3 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
      
        h = self.conv3(g, h)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0
    features = g.ndata['feat']
    features = pca.fit_transform(features)

    labels = g.ndata['label']
    

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
        
    return features, labels
#model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
model = GCN(3, 16, dataset.num_classes)

def plot_3d(data, label):
    np.random.seed(42)
    
    x , y , z = [[],[],[],[],[],[],[]], [[],[],[],[],[],[],[]], [[],[],[],[],[],[],[]]

    for i in range(len(data)): 
        x[labels[i]].append(data[i][0])
        y[labels[i]].append(data[i][1])
        z[labels[i]].append(data[i][2])


    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[0],y[0],z[0], c="red")
    ax.scatter(x[1],y[1],z[1], c="blue")
    ax.scatter(x[2],y[2],z[2], c="yellow")
    ax.scatter(x[3],y[3],z[3], c="green")
    ax.scatter(x[4],y[4],z[4], c="black")
    ax.scatter(x[5],y[5],z[5], c="pink")
    ax.scatter(x[6],y[6],z[6], c="purple")

    plt.show()

if __name__ == "__main__":
    features_prj, labels = train(g, model)
    print(features_prj)
    plot_3d(features_prj, labels)