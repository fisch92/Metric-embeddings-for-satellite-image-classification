import os
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd
from network.resnext import resnext50_32x4d
from utils.math import Distances
from mxnet.gluon.loss import L2Loss
from mxnet import lr_scheduler

from network.abstractNet import AbstractNet

class QuadrupletNet(AbstractNet):

    '''
        Chen, Weihua, X. Chen, J. Zhang und K. Huang (2017). Beyond triplet
        loss: a deep quadruplet network for person re-identification. In: Proceedings of the
        IEEE Conference on Computer Vision and Pattern Recognition, S. 403–412.
    '''

    def __init__(self, loss=Distances.MSE_Dist, margin=0.5, margin2=0.25, alt_loss=False, **kwargs):
        super(QuadrupletNet, self).__init__(**kwargs)
        self.loss = loss
        self.margin = margin
        self.margin2 = margin2
        self.alt_loss = alt_loss
        

    def record_grad(self, quadruplet_batch, ctx):
        pred_batch, pos_batch, neg_batch, neg2_batch = quadruplet_batch
        pred_batch = nd.array(pred_batch, ctx=ctx)
        pos_batch = nd.array(pos_batch, ctx=ctx)
        neg_batch = nd.array(neg_batch, ctx=ctx)
        neg2_batch = nd.array(neg2_batch, ctx=ctx)
        with autograd.record():
            emb_pred = self.net(pred_batch)
            emb_pos = self.net(pos_batch)
            emb_neg = self.net(neg_batch)
            emb_neg2 = self.net(neg2_batch)

            if self.alt_loss:
                loss1 = nd.exp(nd.sqrt(self.loss(emb_pred, emb_pos)))/(nd.exp(nd.sqrt(self.loss(emb_pred, emb_pos)))+nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg)))+nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg2))))
                loss2 = nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg)))/(nd.exp(nd.sqrt(self.loss(emb_pred, emb_pos)))+nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg)))+nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg2))))
                loss3 = nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg2)))/(nd.exp(nd.sqrt(self.loss(emb_pred, emb_pos)))+nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg)))+nd.exp(nd.sqrt(self.loss(emb_pred, emb_neg2))))
                loss = self.loss(loss1, loss2-nd.ones(pred_batch.shape[0], ctx=ctx)) + self.loss(loss1, loss3-nd.full(pred_batch.shape[0], self.margin2, ctx=ctx))
            else:
                loss1 = self.loss(emb_pred, emb_pos) - self.loss(emb_pred, emb_neg) + nd.full(pred_batch.shape[0], self.margin, ctx=ctx)
                loss2 = self.loss(emb_pred, emb_pos) - self.loss(emb_neg2, emb_neg) + nd.full(pred_batch.shape[0], self.margin2, ctx=ctx)

                loss = nd.relu(loss1) + nd.relu(loss2)

        loss.backward()

        return loss.asnumpy()