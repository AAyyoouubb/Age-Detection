import torch
import torchvision
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


class Ndf(LightningModule):

    def __init__(self, depth, n_classes):
        '''
        :param depth: The number of inner nodes in the tree;
        :param n_classes: The number of classes;
        '''
        super(Ndf, self).__init__()

        self.googlenet = torchvision.models.googlenet(pretrained=True, progress=True)
        self.depth = depth
        self.n_nodes = 2 ** depth
        self.fc = nn.Linear(1024, self.n_nodes)  # TODO:1024
        self.pis = nn.Parameter(torch.rand(n_classes, depth + 1))

    def forward(self, x):
        x = self.googlenet(x)
        fn = torch.sigmoid(x)
        P = self.calculate_proba_pathes(fn)
        return P

    def training_step(self, batch, batch_idx):
        # y is 1D tensor containing the corresponding label: starts from 0;
        x, y = batch
        P = self(x)
        return torch.mean(-torch.log(P[list(enumerate(y))]))

    def calculate_proba_pathes(self, fn: torch.Tensor):
        # fn: a matrix where each row represents an input
        probas = torch.zeros(fn.size())
        probas[self.n_nodes // 2] = fn[self.n_nodes // 2]
        for depth in range(1, self.depth):
            # Explore each node in that depth
            # Deal with both children of each parent node simultaneously
            dad_indx = self.nodes // 2 ** depth
            for node in range(0, 2 ** depth, 2):
                # Calculation of children'indices in 'fn' & 'proba'
                tmp = self.nodes // 2 ** (depth + 1)
                left_child_indx = tmp + node * (dad_indx - tmp)
                right_child_indx = tmp + (node + 1) * (dad_indx - tmp)
                # Probability to go to each child
                probas[:, left_child_indx] = probas[:, dad_indx] * fn[:, dad_indx]
                probas[:, right_child_indx] = probas[:, dad_indx] * (1 - fn[:, dad_indx])

        # Calculate the probability for reach each leaf
        depth = self.depth
        proba_leaf = torch.zeros(fn.size()[0], 2 ** depth)
        dad_indx = self.nodes // 2 ** depth
        for node in range(0, 2 ** depth, 2):
            # Calculation of children'indices in 'fn' & 'proba'
            tmp = self.nodes // 2 ** (depth + 1)
            left_child_indx = tmp + node * (dad_indx - tmp)
            right_child_indx = tmp + (node + 1) * (dad_indx - tmp)
            # Probability to go to each child
            proba_leaf[:, left_child_indx] = probas[:, dad_indx] * fn[:, dad_indx]
            proba_leaf[:, right_child_indx] = probas[:, dad_indx] * (1 - fn[:, dad_indx])

        # Finally, calculate the expectation of each label
        # res: a matrix
        res = torch.zeros(proba_leaf.size(0), self.pis.size(0))
        for input in range(res.size(0)):
            res[input] = (self.pis * proba_leaf[input]).sum(dim=0)

        return res
