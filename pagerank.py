import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt

N = 100
DAMP = 0.85
K = 10


def pagerank_message_func(edges):
    return {'pv': edges.src['pv'] / edges.src['deg']}


def pagerank_reduce_func(nodes):
    msg = torch.sum(nodes.mailbox['pv'], dim=1)
    pv = (1 - DAMP) / N + DAMP * msg
    return {'pv': pv}


def naive():
    g = nx.nx.erdos_renyi_graph(N, 0.1)
    g = dgl.DGLGraph(g)
    nx.draw(g.to_networkx(), node_size=50, node_color=[[.6, .6, .6]])
    plt.show()

    g.ndata['pv'] = torch.ones(N) / N
    g.ndata['deg'] = g.out_degrees(g.nodes()).float()

    g.register_message_func(pagerank_message_func)
    g.register_reduce_func(pagerank_reduce_func)

    for u, v in zip(*g.edges()):
        g.send((u, v))
    for v in g.nodes():
        g.recv(v)


def pagerank_batch(g):
    g.send(g.edges())
    g.recv(g.nodes())


def pagerank_level2(g):
    g.update_all()


if __name__ == '__main__':
    naive()
