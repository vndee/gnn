import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

from dgl import DGLGraph
from dgl.contrib.data import load_data
from functools import partial


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None, activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))

        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

        if self.bias:
            nn.init.xavier_normal_(self.bias, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)

            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_hidden_layers=1):
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        self.layers = nn.ModuleList()
        self.layers.append(RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases, activation=F.relu, is_input_layer=True))
        for _ in range(self.num_hidden_layers):
            self.layers.append(RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, activation=F.relu))
        self.layers.append(RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases, activation=partial(F.softmax, dim=1)))

        self.features = torch.arange(self.num_nodes)

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')


if __name__ == '__main__':
    data = load_data(dataset='aifb')
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes
    labels = data.labels
    train_idx = data.train_idx
    val_idx = train_idx[:len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]

    edge_type = torch.from_numpy(data.edge_type)
    edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)

    labels = torch.from_numpy(labels).view(-1)

    n_hidden = 16
    n_bases = -1
    n_hidden_layers = 0
    n_epochs = 25
    lr = 0.01
    l2norm = 0

    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(data.edge_src, data.edge_dst)
    g.edata.update({'rel_type': edge_type, 'norm': edge_norm})

    model = RGCN(len(g), n_hidden, num_classes, num_rels, num_bases=n_bases, num_hidden_layers=n_hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

    print('Start training...')
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(g)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()

        optimizer.step()

        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
        train_acc = train_acc.item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx])
        val_acc = val_acc.item() / len(val_idx)

        print(f'Epoch {epoch} | Train Accuracy: {round(train_acc, 4)} | Train Loss: {round(loss.item(), 4)} '
              f'| Validation Accuracy: {round(val_acc, 4)} | Validation Loss: {round(val_loss.item(), 4)}')
