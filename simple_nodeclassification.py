import dgl
import time
import torch
import networkx
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.animation as animation


def build_karate_club_graph():
    g = dgl.DGLGraph()
    g.add_nodes(34)
    edge_list = [
        (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)
    ]

    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.ndata['feat'] = torch.eye(34)

    return g


def gcn_message(edges):
    return {'msg': edges.src['h']}


def gcn_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        g.ndata['h'] = inputs
        g.send(g.edges(), gcn_message)
        g.recv(g.nodes(), gcn_reduce)
        h = g.ndata.pop('h')
        return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, _inputs):
        h = self.gcn1(g, _inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h


def visualize_network(G):
    networkx_G = G.to_networkx().to_undirected()
    pos = networkx.kamada_kawai_layout(networkx_G)
    networkx.draw(networkx_G, pos, with_labels=True, node_color=[[.5, .7, .7]])
    plt.show()


def draw(i, G, ax):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos, colors = {}, []

    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)

    ax.cla()
    ax.axis('off')
    ax.set_title(f'Epoch: {i}')

    networkx_G = G.to_networkx().to_undirected()
    networkx.draw_networkx(networkx_G.to_undirected(), pos, node_color=colors, with_labels=True, node_size=300, ax=ax)


if __name__ == '__main__':
    G = build_karate_club_graph()
    net = GCN(34, 5, 2)

    inputs = torch.eye(34)
    labeled_nodes = torch.tensor([0, 33])
    labels = torch.tensor([0, 1])

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    all_logits = []

    for epoch in range(30):
        logits = net(G, inputs)
        all_logits.append(logits.detach())
        log = F.log_softmax(logits, 1)
        loss = F.nll_loss(log[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch} | Loss: {round(loss.item(), 4)}')

    fig = plt.figure(dpi=500)
    fig.clf()
    ax = fig.subplots()
    draw(29, G, ax)
    plt.show()
