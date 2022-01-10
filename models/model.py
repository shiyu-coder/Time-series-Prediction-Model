import torch
import math
import numpy as np
import torch.nn as nn
from models.decoder import DecoderLayer
from models.attention import ProbAttention, AttentionLayer, FullAttention
from models.decoder import Decoder
from utils.positionalEncoding import PositionalEncoding
from models.embed import DataEmbedding
from models.encoder import DFCE


class DenseFormer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=4, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, L_d=64,
                 device=torch.device('cuda:0')):
        super(DenseFormer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = DFCE(factor, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 output_attention, distil, L_d=None).double()
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ).double()
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    X = torch.rand(size=(32, 64, 7), dtype=torch.double)
    dropout = 0.0
    d_model = 512
    attn = 'prob'
    d_ff = 512
    enc_in = 7
    dec_in = 7
    c_out = 1
    seq_len = 64
    label_len = 64
    out_len = 1
    factor = 5
    n_heads = 8
    Attn = FullAttention
    e_layers = 4
    activation = 'gelu'
    distil = True
    output_attention = False
    net = DenseFormer(enc_in, dec_in, c_out, seq_len, label_len, out_len).double()

    X = net(X, X, X, X)
    print(X.shape)



