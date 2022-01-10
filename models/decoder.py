import torch.cuda
import torch.nn as nn
from models.attention import MultiHeadAttention, AttentionLayer, FullAttention
from utils.addNorm import AddNorm
import torch.nn.functional as F


# class DecoderLayer(nn.Module):
#     def __init__(self, key_size=32, num_hiddens=128, num_heads=4, seq_len=4, drop_out=0.1):
#         super(DecoderLayer, self).__init__()
#
#         self.dense1 = nn.Linear(num_hiddens, num_hiddens)
#         self.dense2 = nn.Linear(num_hiddens, num_hiddens)
#         self.at1 = MaskedMultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads)
#         self.at2 = MaskedMultiHeadAttention(key_size, num_hiddens, key_size, num_hiddens, num_heads)
#         self.norm1 = AddNorm((seq_len, num_hiddens), drop_out)
#         self.norm2 = AddNorm((seq_len, num_hiddens), drop_out)
#         self.norm3 = AddNorm((seq_len, num_hiddens), drop_out)
#         self.actFun = nn.ELU()
#         self.drop = nn.Dropout(drop_out)
#
#     def forward(self, x, cross):
#         x = self.norm1(x, self.at1(x, x, x))
#         x = self.norm2(x, self.at2(x, cross, cross))
#
#         x = self.norm3(x, self.actFun(self.dense2(self.drop(self.actFun(self.dense1(x))))))
#
#         return x
#
#
# class Decoder(nn.Module):
#     def __init__(self, layer_num=2, key_size=32, num_hiddens=128, num_heads=4, seq_len=4, drop_out=0.1):
#         super(Decoder, self).__init__()
#         self.layer_num = layer_num
#         self.dec = []
#         if torch.cuda.is_available():
#             for i in range(layer_num):
#                 self.dec.append(DecoderLayer(key_size, num_hiddens, num_heads, seq_len, drop_out).cuda())
#         else:
#             for i in range(layer_num):
#                 self.dec.append(DecoderLayer(key_size, num_hiddens, num_heads, seq_len, drop_out))
#
#     def forward(self, x, cross):
#         for i in range(self.layer_num):
#             x = self.dec[i](x, cross)
#
#         return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


if __name__ == '__main__':
    X = torch.rand(size=(8, 256, 512))
    ams = torch.rand(size=(8, 512*8, 512))
    dropout = 0.0
    d_model = 512
    attn = 'prob'
    d_ff = 512
    factor = 5
    n_heads = 8
    Attn = FullAttention
    d_layers = 6
    activation = 'gelu'
    distil = True
    output_attention = False
    decoder = Decoder(
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
    )
    dec_out = decoder(X, ams)
    print(dec_out.shape)
