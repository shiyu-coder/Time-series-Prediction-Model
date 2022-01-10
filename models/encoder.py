import torch.nn as nn
import torch
from models.attention import ProbAttention, AttentionLayer, FullAttention, MultiHeadAttention
from utils.addNorm import AddNorm
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  padding_mode='circular').double()
        self.norm = nn.BatchNorm1d(c_in).double()
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        if torch.cuda.is_available():
            self.downConv = self.downConv.cuda()
            self.norm = self.norm.cuda()
            self.maxPool = self.maxPool.cuda()

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1).double()
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1).double()
        self.norm1 = nn.LayerNorm(d_model).double()
        self.norm2 = nn.LayerNorm(d_model).double()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.norm1 = self.norm1.cuda()
            self.norm2 = self.norm2.cuda()

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x+y), attn


class FCEncoder(nn.Module):
    def __init__(self, front_attn_layers, back_attn_layers, front_conv_layers=None,
                 back_conv_layers=None, norm_layer=None):
        super(FCEncoder, self).__init__()
        self.front_attn_layers = nn.ModuleList(front_attn_layers).double()
        self.front_conv_layers = nn.ModuleList(front_conv_layers).double() if front_conv_layers is not None else None
        self.back_attn_layers = nn.ModuleList(back_attn_layers).double()
        self.back_conv_layers = nn.ModuleList(back_conv_layers).double() if back_conv_layers is not None else None
        self.norm = norm_layer.double()
        if torch.cuda.is_available():
            self.front_attn_layers = self.front_attn_layers.cuda()
            self.front_conv_layers = self.front_conv_layers.cuda()
            self.back_attn_layers = self.back_attn_layers.cuda()
            self.back_conv_layers = self.back_conv_layers.cuda()
            self.norm = self.norm.cuda()

    def forward(self, x, attn_mask=None):
        mid_outputs = []
        if self.front_conv_layers is not None:
            for attn_layer, conv_layer in zip(self.front_attn_layers, self.front_conv_layers):
                mid_output, attn = attn_layer(x, attn_mask=attn_mask)
                mid_output = conv_layer(mid_output)
                mid_outputs.append(mid_output)
                # print('tt', mid_output.shape)
                # print('tmp', mid_output.shape)
        else:
            for attn_layer in self.front_attn_layers:
                mid_output, attn = attn_layer(x, attn_mask=attn_mask)
                mid_outputs.append(mid_output)

        mid_attn_map = torch.cat(mid_outputs, 1)
        final_outputs = []

        if self.back_conv_layers is not None:
            for attn_layer, conv_layer in zip(self.back_attn_layers, self.back_conv_layers):
                final_output, attn = attn_layer(mid_attn_map, attn_mask=attn_mask)
                final_output = conv_layer(final_output)
                final_outputs.append(final_output)
                # print('tmp', final_output.shape)
        else:
            for attn_layer in self.back_attn_layers:
                final_output, attn = attn_layer(mid_attn_map, attn_mask=attn_mask)
                final_outputs.append(final_output)

        final_attn_map = torch.cat(final_outputs, 1)

        if self.norm is not None:
            final_attn_map = self.norm(final_attn_map)
        return final_attn_map


class DFCE(nn.Module):

    def __init__(self, factor=5, d_model=512, n_heads=8, e_layers=4, d_ff=512, dropout=0.0, attn='prob',
                 activation='gelu', output_attention=False, distil=True, L_d=None):
        super(DFCE, self).__init__()
        self.L_d = L_d
        if L_d is not None:
            self.conv1 = nn.Conv1d(in_channels=L_d*8, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=L_d, kernel_size=1)
            self.dropout = nn.Dropout(dropout)
            self.activation = F.relu if activation == "relu" else F.gelu
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.FCEs = nn.Sequential(*[FCEncoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(e_layers)
            ],
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model,
                ) for l in range(e_layers)
            ] if distil else None,
            [
                ConvLayer(
                    d_model,
                ) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        ) for i in range(3)])

    def forward(self, x, attn_mask=None):
        x1 = self.FCEs[0](x, attn_mask)
        inp1 = torch.cat((x, x1), dim=1)
        x2 = self.FCEs[1](inp1, attn_mask)
        inp2 = torch.cat((x, x1, x2), dim=1)
        x3 = self.FCEs[2](inp2, attn_mask)
        inp3 = torch.cat((x, x1, x2, x3), dim=1)
        if self.L_d is not None:
            y = self.dropout(self.activation(self.conv1(inp3)))
            y = self.dropout(self.conv2(y))
        else:
            y = inp3
        return y


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        ams = []
        begin = True
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                print('tmp', x.shape)
                x = conv_layer(x)
                if begin:
                    begin = False
                else:
                    ams.append(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            ams.append(x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
                ams.append(x)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, ams


class SingleEncoder(nn.Module):

    def __init__(self, layer_num=2, num_hiddens=128, num_heads=4, seq_len=4, drop_out=0.1, min_output_size=32):
        super(SingleEncoder, self).__init__()
        self.layer_num = layer_num

        self.mha = []
        self.an = []
        self.li = []

        seq_size = num_hiddens
        if torch.cuda.is_available():
            for i in range(layer_num):
                self.mha.append(MultiHeadAttention(seq_size, seq_size, seq_size, seq_size, num_heads, drop_out).cuda())
                self.an.append(AddNorm((seq_len, seq_size), drop_out).cuda())
                self.li.append(nn.Linear(seq_size, max(min_output_size, seq_size//2)).cuda())
                seq_size = max(min_output_size, seq_size//2)
        else:
            for i in range(layer_num):
                self.mha.append(MultiHeadAttention(seq_size, seq_size, seq_size, seq_size, num_heads, drop_out))
                self.an.append(AddNorm((seq_len, seq_size), drop_out))
                self.li.append(nn.Linear(seq_size, max(min_output_size, seq_size//2)))
                seq_size = max(min_output_size, seq_size//2)

        self.actFun = nn.ELU()
        self.key_size = seq_size

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.an[i](x, self.mha[i](x, x, x))
            x = self.actFun(self.li[i](x))
        return x


class SALayer(nn.Module):

    def __init__(self, layer_num=2, c1=128, c2=1, num_heads=4, seq_len=4, drop_out=0.1):
        super(SALayer, self).__init__()
        self.layer_num = layer_num

        self.mha = []
        self.an = []
        self.li = []

        channels = [c1, 64, 32, 16, c2]

        if torch.cuda.is_available():
            for i in range(layer_num):
                self.mha.append(MultiHeadAttention(channels[i], channels[i], channels[i],
                                                   channels[i], num_heads, drop_out).cuda())
                self.an.append(AddNorm((seq_len, channels[i]), drop_out).cuda())
                self.li.append(nn.Linear(channels[i], channels[i+1]).cuda())
        else:
            for i in range(layer_num):
                self.mha.append(MultiHeadAttention(channels[i], channels[i], channels[i],
                                                   channels[i], num_heads, drop_out))
                self.an.append(AddNorm((seq_len, channels[i]), drop_out))
                self.li.append(nn.Linear(channels[i], channels[i + 1]))

        self.actFun = nn.ELU()

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.an[i](x, self.mha[i](x, x, x))
            x = self.actFun(self.li[i](x))
        return x


if __name__ == '__main__':
    X = torch.rand(size=(8, 32, 512))
    dropout = 0.0
    d_model = 512
    attn = 'prob'
    d_ff = 512
    factor = 5
    n_heads = 8
    Attn = FullAttention
    e_layers = 4
    activation = 'gelu'
    distil = True
    output_attention = False
    # encoder = Encoder(
    #     [
    #         EncoderLayer(
    #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
    #                            d_model, n_heads),
    #             d_model,
    #             d_ff,
    #             dropout=dropout,
    #             activation=activation
    #         ) for l in range(e_layers)
    #     ],
    #     [
    #         ConvLayer(
    #             d_model
    #         ) for l in range(e_layers - 1)
    #     ] if distil else None,
    #     norm_layer=torch.nn.LayerNorm(d_model)
    # )

    # encoder = FCEncoder(
    #     [
    #         EncoderLayer(
    #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
    #                            d_model, n_heads),
    #             d_model,
    #             d_ff,
    #             dropout=dropout,
    #             activation=activation
    #         ) for l in range(e_layers)
    #     ],
    #     [
    #         EncoderLayer(
    #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
    #                            d_model, n_heads),
    #             d_model,
    #             d_ff,
    #             dropout=dropout,
    #             activation=activation
    #         ) for l in range(e_layers)
    #     ],
    #     [
    #         ConvLayer(
    #             d_model
    #         ) for l in range(e_layers)
    #     ] if distil else None,
    #     [
    #         ConvLayer(
    #             d_model
    #         ) for l in range(e_layers)
    #     ] if distil else None,
    #     norm_layer=torch.nn.LayerNorm(d_model)
    # )

    encoder = DFCE()

    X = encoder(X)
    print(X.shape)
    # for am in ams:
    #     print(am.shape)

