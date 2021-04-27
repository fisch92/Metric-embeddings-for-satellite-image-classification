import os
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from network.resnext import resnext50_32x4d


class AbstractNet():

    def __init__(self, name='net', ctx=[mx.gpu()], learning_rate=0.0002, net=resnext50_32x4d(ctx=mx.gpu()), load=True, **kwargs):
        self.name = name
        self.ctx = ctx
        self.net = net
        if load:
            self.load()
        else:
            self.reset()
        self.trainer = gluon.Trainer(self.net.collect_params(), mx.optimizer.Adam(learning_rate=learning_rate))

        for p in self.net.collect_params().values():
            p.grad_req = 'add'
        

    def predict(self, batch):
        if not isinstance(batch, nd.NDArray):
            batch = nd.array(batch, ctx=self.ctx[0])
        out = self.net(batch)
        return out


    def train_step(self, triplet_batch):
        counter = 0
        for cctx in self.ctx[::-1]:
            cBatch = []
            for part in triplet_batch:
                part = part[counter::len(self.ctx)]
                cBatch.append(part)
            loss = self.record_grad(cBatch, ctx=cctx)
            counter += 1

        self.trainer.step(triplet_batch[0].shape[0], ignore_stale_grad=True)
        for p in self.net.collect_params().values():
            p.zero_grad()

        return loss

    def record_grad(self, triplet_batch, ctx):
        return NotImplemented

    def save(self):
        self.net.save_parameters(self.name)

    def load(self, init=mx.init.Xavier()):
        if os.path.exists(self.name):
            print('load: ', self.name)
            self.net.load_parameters(self.name, ctx=[mx.cpu()]+self.ctx)
        else:
            self.net.initialize(init=init, ctx=[mx.cpu()]+self.ctx)

    def reset(self, init=mx.init.Xavier(), lr=0.0002):
        print("reset", lr)
        self.trainer = gluon.Trainer(self.net.collect_params(), mx.optimizer.Adam(learning_rate=lr))
        
        self.net.initialize(init=init, ctx=self.ctx)
