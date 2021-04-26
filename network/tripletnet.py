import os
import random
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd
from network.resnext import resnext50_32x4d
from utils.math import Distances
from mxnet.gluon.loss import L2Loss
from mxnet import lr_scheduler

from network.abstractNet import AbstractNet

class TripletNet(AbstractNet):

    '''
        Wang, Jiang, Y. Song, T. Leung, C. Rosenberg, J. Wang, J. Philbin,
        B. Chen und Y. Wu (2014). Learning fine-grained image similarity with deep
        ranking. In: Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition, S. 1386–1393.

        Hoffer, Elad und N. Ailon (2015). Deep metric learning using triplet net-
        work . In: International Workshop on Similarity-Based Pattern Recognition, S. 84–
        92. Springer.

        Jean, Neal, S. Wang, A. Samar, G. Azzari, D. Lobell und S. Ermon
        (2019). Tile2Vec: Unsupervised representation learning for spatially distributed data.
        In: Proceedings of the AAAI Conference on Artificial Intelligence, Bd. 33, S. 3967–
        3974.
    '''

    def __init__(self, loss=Distances.MSE_Dist, margin=0.5, alt_loss=False, **kwargs):
        super(TripletNet, self).__init__(**kwargs)
        
        self.loss = loss
        self.alt_loss = alt_loss
        self.margin = margin

    def record_grad(self, triplet_batch, ctx):
        pred_batch, pos_batch, neg_batch = triplet_batch
        pred_batch = nd.array(pred_batch, ctx=ctx)
        pos_batch = nd.array(pos_batch, ctx=ctx)
        neg_batch = nd.array(neg_batch, ctx=ctx)
        
        with autograd.record():

            pred = self.net(nd.concat(pred_batch, pos_batch, neg_batch, dim=0))
            emb_pred, emb_pos, emb_neg = nd.split(pred, 3, axis=0)

            if self.alt_loss:
                loss1 = nd.exp(nd.sqrt(self.loss(emb_pred, emb_pos)))/(nd.exp(nd.sqrt(self.loss(emb_pred, emb_pos)))+nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg))))
                loss2 = nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg)))/(nd.exp(nd.sqrt(self.loss(emb_pred, emb_pos)))+nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg))))
                loss = self.loss(loss1, loss2-nd.ones(pred_batch.shape[0], ctx=ctx))
            else:
                loss = self.loss(emb_pred, emb_pos) - self.loss(emb_pred, emb_neg) + nd.full(pred_batch.shape[0], self.margin, ctx=ctx)
            loss = nd.relu(loss)

        loss.backward()
            
        return loss