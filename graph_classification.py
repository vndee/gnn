import dgl
import torch
import networkx as nx
import torch.nn as nn
import dgl.function as fn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dgl.data import MiniGCDataset
from torch.utils.data import DataLoader

dataset = MiniGCDataset(80, 10, 20)
graph, label = dataset[0]
fig, ax = plt.subplots()
nx.draw(graph.to_networkx(), ax=ax)
ax.set_title(f'Class: {label}')
plt.show()


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


msg = fn.copy_src(src='h', out='m')


def reduce(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)
        ])

        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


if __name__ == '__main__':
    train_set = MiniGCDataset(320, 10, 20)
    test_set = MiniGCDataset(80, 10, 20)
    data_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate)

    model = Classifier(1, 256, train_set.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    epoch_losses = []
    for epoch in range(80):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

        print(f'Epoch {epoch}, loss: {epoch_loss}')
        epoch_losses.append(epoch_loss)

    plt.title('Cross entropy averaged over minibatches')
    plt.plot(epoch_losses)
    plt.show()

    model.eval()
    test_X, test_Y = map(list, zip(*test_set))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)

    print(f'Accuracy of sampled predictions on the test set: {(test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100}')
    print(f'Accuracy of argmax predictions on the test set: {(test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100}')
