from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd
import numpy as np
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, Sequential,\
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout, HybridLambda, InstanceNorm, SELU, LayerNorm, AvgPool3D, PReLU

from network.vit.tiler import Tiler,DeTiler
from gluonnlp.model import MultiHeadAttentionCell, DotProductAttentionCell


class TransformerNet(HybridBlock):

    '''
        Vaswani, Ashish, et al. "Attention is all you need." arXiv preprint arXiv:1706.03762 (2017).

        https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''
    def __init__(self, img_size, size, in_channel=3, num_heads=8, emb_size=(256+128), classes=1, norm_output=True):
        super(TransformerNet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.img_size = img_size
        self.norm_output = norm_output

        #self.down1 = DownBlock(16, inputNorm=False)


        self.tiler1 = Tiler((img_size)//size, 1, emb_size)
        self.proj1 = Conv2D(emb_size, kernel_size=self.size, strides=self.size, use_bias=False)

        self.classifier = nn.HybridSequential()
        self.classifier.add(nn.LayerNorm(in_channels=emb_size))
        self.classifier.add(nn.Dense(classes))
        
        self.encoder = TransformerEncoderSequence(emb_size=emb_size, num_heads=num_heads, drop_p=0.2, forward_expansion=4, forward_drop_p=0.2, prefix='enc_', depth=10)
        #self.decoder = TransformerDecoderSequence(emb_size=emb_size, num_heads=num_heads, drop_p=0.2, forward_expansion=4, forward_drop_p=0.2, prefix='dec_', depth=1)
        if norm_output:
            self.norm = LayerNorm()


    def hybrid_forward(self, F, x):
        #print(x.shape)
        proj1 = self.proj1(x)
        emb1 = self.tiler1(proj1)
        y = self.encoder(emb1)
        #y = self.decoder(y, y1)
        #y = self.norm(y)

        y = self.classifier(y)
        if self.norm_output:
            y = self.norm(y)
        #print(y.shape)        
        return y

class TransformerEncoderSequence(HybridBlock):
    def __init__(self,
                 emb_size = 768,
                 drop_p = 0.0,
                 forward_expansion = 4,
                 forward_drop_p: float = 0.0,
                 num_heads = 8,
                 depth = 1,
                 prefix = None,
                 ** kwargs):
        super(TransformerEncoderSequence, self).__init__()

        self.encoder = nn.HybridSequential(prefix='enc_')
        for depth in range(0, depth):
            encoder = TransformerEncoderBlock(emb_size=emb_size, num_heads=num_heads, drop_p=drop_p, forward_expansion=forward_expansion, forward_drop_p=forward_drop_p, prefix='enc_'+str(depth)+'_'+prefix, inputNorm=(depth!=0))
            self.encoder.add(encoder)

    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = x[:, 0]
        return x

class TransformerDecoderSequence(HybridBlock):
    def __init__(self,
                 emb_size = 768,
                 drop_p = 0.0,
                 forward_expansion = 4,
                 forward_drop_p: float = 0.0,
                 num_heads = 8,
                 depth = 1,
                 prefix = None,
                 ** kwargs):
        super(TransformerDecoderSequence, self).__init__()

        self.embNorm = nn.LayerNorm(in_channels=emb_size)
        self.decoder = []
        for depth in range(0, depth):
            decoder = TransformerDecoderBlock(emb_size=emb_size, num_heads=num_heads, drop_p=drop_p, forward_expansion=forward_expansion, forward_drop_p=forward_drop_p, prefix='dec'+str(depth)+'_'+prefix, inputNorm=(depth!=0))
            self.decoder.append(decoder)
            self.register_child(decoder)

    def hybrid_forward(self, F, x, target):
        x = self.embNorm(x)
        for dec in self.decoder:
            target = dec(target, x)
            #print(F.moments(target, axes=(0,1,2)))


        return target

class TransformerDecoderBlock(HybridBlock):
    def __init__(self,
                 emb_size = 768,
                 drop_p = 0.0,
                 forward_expansion = 4,
                 forward_drop_p: float = 0.0,
                 num_heads = 8,
                 prefix = None,
                 inputNorm = False,
                 ** kwargs):
        super(TransformerDecoderBlock, self).__init__()
        self.inputNorm = inputNorm
        if inputNorm:
            self.inputNorm = nn.LayerNorm(in_channels=emb_size)
        self.attNorm = nn.LayerNorm(in_channels=emb_size)
        self.outputNorm = nn.LayerNorm(in_channels=emb_size)
        self.selfAtt = MultiHeadAttentionCell(
            DotProductAttentionCell(units=emb_size, scaled=True, dropout=drop_p, use_bias=False, prefix='decSelfAtt_'+prefix, normalized=False),
            query_units=emb_size, 
            key_units=emb_size, 
            value_units=emb_size, 
            num_heads=num_heads,
            use_bias=False
        )
        self.encAtt = MultiHeadAttentionCell(
            DotProductAttentionCell(units=emb_size, scaled=True, dropout=drop_p, use_bias=False, prefix='decEncAtt_'+prefix, normalized=False),
            query_units=emb_size, 
            key_units=emb_size, 
            value_units=emb_size, 
            num_heads=num_heads,
            use_bias=False
        )
        self.ff = nn.HybridSequential()
        self.ff.add(nn.Dense(emb_size*forward_expansion, flatten=False, use_bias=False))
        self.ff.add(nn.GELU())
        self.ff.add(nn.Dropout(forward_drop_p))
        self.ff.add(nn.Dense(emb_size, flatten=False, use_bias=False))
        self.ff.add(nn.Dropout(forward_drop_p))

    def hybrid_forward(self, F, target, emb):
        if self.inputNorm:
            att = self.inputNorm(target)
        else:
            att = target
        att,_ = self.selfAtt(att, att)
        att = att + target

        y = self.attNorm(att)
        y,_ = self.encAtt(y, emb, value=emb)
        att = att + y

        y = self.outputNorm(att)
        y = self.ff(y)

        y = (y+att)

        return y

class TransformerEncoderBlock(HybridBlock):
    def __init__(self,
                 emb_size = 768,
                 drop_p = 0.0,
                 forward_expansion = 4,
                 forward_drop_p: float = 0.0,
                 num_heads = 8,
                 prefix = None,
                 inputNorm = False,
                 ** kwargs):
        super(TransformerEncoderBlock, self).__init__()
        self.inputNorm = inputNorm
        if inputNorm:
            self.inputNorm = nn.LayerNorm(in_channels=emb_size)
        self.outputNorm = nn.LayerNorm(in_channels=emb_size)
        self.encoder = MultiHeadAttentionCell(
            DotProductAttentionCell(units=emb_size, scaled=True, dropout=drop_p, use_bias=False, prefix='encSelfAtt_'+prefix, normalized=False),
            query_units=emb_size, 
            key_units=emb_size, 
            value_units=emb_size, 
            num_heads=num_heads,
            use_bias=False
        )
        self.ff = nn.HybridSequential()
        self.ff.add(nn.Dense(emb_size*forward_expansion, flatten=False, use_bias=False))
        self.ff.add(nn.GELU())
        self.ff.add(nn.Dropout(forward_drop_p))
        self.ff.add(nn.Dense(emb_size, flatten=False, use_bias=False))
        self.ff.add(nn.Dropout(forward_drop_p))

    def hybrid_forward(self, F, x):
        if self.inputNorm:
            att = self.inputNorm(x)
        else:
            att = x
        att,_ = self.encoder(att, att)
        att = att + x
        y = self.outputNorm(att)
        y = self.ff(y)

        y = (y+att)

        return y

class UpBlock(HybridBlock):
    

    def __init__(self, channels, inputNorm=True,
                 downsample=False, norm_layer=InstanceNorm, norm_kwargs=None, in_channels=0, image_size=128, **kwargs):
        super(UpBlock, self).__init__(**kwargs)
        
        with self.name_scope():
            self.in_channels = in_channels
            self.body = nn.HybridSequential(prefix='')
            self.body.add(Conv2DTranspose(channels, kernel_size=1, use_bias=False))
            if inputNorm:
                self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.body.add(nn.Activation('relu'))
            self.body.add(Conv2DTranspose(channels, kernel_size=4, strides=2, padding=1, use_bias=False, in_channels=in_channels))
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.body.add(nn.Activation('relu'))
            self.body.add(Conv2DTranspose(channels, kernel_size=1, use_bias=False, in_channels=channels))

            self.downsample = nn.HybridSequential(prefix='')
            
            self.downsample.add(Conv2DTranspose(channels, kernel_size=1, use_bias=False))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))#BatchNorm(momentum=MOMENTUM, use_global_stats=GLOBAL_STATS))
            

            

    def hybrid_forward(self, F, x):
        
        residual = x
        att = x
        x = self.body(x)


        if self.downsample:
            residual = F.contrib.BilinearResize2D(residual, scale_height=2, scale_width=2, mode='odd_scale')
            residual = self.downsample(residual)
            
        
        
        x = x + residual

        return x

class DownBlock(HybridBlock):
    

    def __init__(self, channels, inputNorm=True,
                 norm_layer=InstanceNorm, norm_kwargs=None, in_channels=0, **kwargs):
        super(DownBlock, self).__init__(**kwargs)
        
        with self.name_scope():
            self.in_channels = in_channels
            self.channels = channels
            #print(group_width, D, cardinality, stride)

            self.body = nn.HybridSequential(prefix='')
            self.body.add(Conv2D(channels, kernel_size=1, use_bias=False))
            if inputNorm:
                self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.body.add(nn.LeakyReLU(0.2))
            self.body.add(Conv2D(channels, kernel_size=4, strides=2, padding=1, use_bias=False, in_channels=channels))
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.body.add(nn.LeakyReLU(0.2))
            self.body.add(Conv2D(channels, kernel_size=1, use_bias=False, in_channels=channels))
            self.body.add(nn.Dropout(0.2))

            self.downsample = nn.HybridSequential(prefix='')
            
            self.downsample.add(Conv2D(channels, kernel_size=1, use_bias=False))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))

            

    def hybrid_forward(self, F, x):
        residual = x
        att = x
        x = self.body(x)
        residual = self.downsample(residual)

        residual = F.contrib.BilinearResize2D(residual, scale_height=0.5, scale_width=0.5, mode='odd_scale')
        
        return x+residual