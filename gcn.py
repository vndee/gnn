import dgl
import time
import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dgl import DGLGraph
from dgl.data import citation_graph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.gcn1 = GCNLayer(1433, 16, F.relu)
        self.gcn2 = GCNLayer(16, 7, None)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(x)
        return x


def load_cora_data():
    data = citation_graph.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = data.graph

    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())

    figs, ax = plt.subplots()
    nx.draw(g.to_networkx(), ax=ax)
    ax.set_title('Cora citation graph')
    plt.show()
    return g, features, labels, train_mask, test_mask


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


net = GCNNet()
print(net)

g, features, labels, train_mask, test_mask = load_cora_data()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
dur = []

for epoch in range(50):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    logits = net(g, features)
    log = F.log_softmax(logits, 1)
    loss = F.nll_loss(log[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    acc = evaluate(net, g, features, labels, test_mask)
    print(f'Epoch: {epoch} | Loss: {round(loss.item(), 4)} | Accuracy: {round(acc, 4)} | Time(s): {np.mean(dur)}')
