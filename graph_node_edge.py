import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt


def dgl_graph_from_networkx():
    g_nx = nx.petersen_graph()
    g_dgl = dgl.DGLGraph(g_nx)
    plt.subplot(121)
    nx.draw(g_nx, with_labels=True)
    plt.subplot(122)
    nx.draw(g_dgl.to_networkx(), with_labels=True)
    plt.show()


def dgl_node_edge():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    for i in range(1, 4):
        g.add_edge(i, 0)

    src = list(range(5, 8))
    dst = [0]*3
    g.add_edges(src, dst)

    src = torch.tensor([8, 9])
    dst = torch.tensor([0, 0])
    g.add_edges(src, dst)

    g.clear()
    g.add_nodes(10)
    src = torch.tensor(list(range(1, 10)))
    g.add_edges(src, 0)

    x = torch.randn(10, 3)
    g.ndata['x'] = x
    g.nodes[0].data['x'] = torch.zeros(1, 3)
    g.nodes[[0, 1, 2]].data['x'] = torch.zeros(3, 3)
    g.nodes[torch.tensor([0, 1, 2])].data['x'] = torch.zeros(3, 3)

    g.edges['w'] = torch.randn(9, 2)
    g.edges[1].data['w'] = torch.randn(1, 2)
    g.edges[[0, 1, 2]].data['w'] = torch.zeros(3, 2)
    g.edges[torch.tensor([0, 1, 2])].data['w'] = torch.zeros(3, 2)

    g.edges[1, 0].data['w'] = torch.ones(1, 2)
    g.edges[[1, 2, 3], [0, 0, 0]].data['w'] = torch.ones(3, 2)

    nx.draw(g.to_networkx(), with_labels=True)
    plt.show()


def multigraph():
    g_multi = dgl.DGLGraph(multigraph=True)
    g_multi.add_nodes(10)
    g_multi.ndata['x'] = torch.randn(10, 2)

    g_multi.add_edges(list(range(1, 10)), 0)
    g_multi.add_edge(1, 0)

    g_multi.edata['w'] = torch.randn(10, 2)
    g_multi.edges[1].data['w'] = torch.zeros(1, 2)


if __name__ == '__main__':
    dgl_node_edge()
