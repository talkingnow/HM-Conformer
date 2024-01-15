import torch
import torch.nn as nn
import torch.nn.functional as F

from .wenet.transformer.encoder_mp_nonlpe_hieracls22 import ConformerEncoder
from ..backend.attention import SelfWeightedPooling

class HM_Conformer(nn.Module):
    def __init__(self, bin_size=120, num_blocks=6, output_size=128, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos", use_ssl=False, ssl_layers=12, linear_units = 256, cnn_module_kernel=15,
            downsample_layer=[1,3], pooling_size=2, input_seq_len=200, layer_cls=True, dropout=0, emb_dropout=0, multiloss=False,
            use_emb=[0,1,2,3,4]):

        super(HM_Conformer, self).__init__()
        
        self.use_ssl = use_ssl
        if self.use_ssl:
            self.w = nn.Parameter(torch.rand(1, ssl_layers + 1, 1, 1))
        
        self.conformer_mp = ConformerEncoder(input_size=bin_size, linear_units=linear_units, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type, cnn_module_kernel=cnn_module_kernel,
                downsample_layer=downsample_layer, pooling_size=pooling_size, input_seq_len=input_seq_len, layer_cls=layer_cls)
        
        self.use_emb = use_emb
        
        if 0 in use_emb:
            self.fc0 = nn.Linear(output_size, output_size)
            self.bn0 = nn.BatchNorm1d(output_size)
        if 1 in use_emb:
            self.fc1 = nn.Linear(output_size, output_size)
            self.bn1 = nn.BatchNorm1d(output_size)
        if 2 in use_emb:
            self.fc2 = nn.Linear(output_size, output_size)
            self.bn2 = nn.BatchNorm1d(output_size)
        if 3 in use_emb:
            self.attn = SelfWeightedPooling(output_size, num_head=1, mean_only=True)

        self.fc3 = nn.Linear(int(output_size * (len(use_emb)-1)), output_size)
        self.bn3 = nn.BatchNorm1d(output_size)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
        
        self.multiloss = multiloss
        
    def forward(self, x):
        # (batchsize, length, feature_dim)
        assert len(x.size()) == 3, f'Input size error in Conformer. Need 3, but get {len(x.size())}'
        
        if self.use_ssl:
            # (B, L, T, H)
            # weighted-sum
            x = x * self.w.repeat(x.size(0), 1, 1, 1)
            x = x.sum(dim=1)
        
        lens = torch.ones(x.shape[0]).to(x.device)
        lens = torch.round(lens * x.shape[1]).int()
        x, cls = self.conformer_mp(x, lens)
        B, T, H = x.size()
        
        emb = []
        if 0 in self.use_emb:
            emb.append(self.emb_dropout(self.bn0(self.silu(self.fc0(cls[:, 0, :])))))
        if 1 in self.use_emb:
            emb.append(self.emb_dropout(self.bn1(self.silu(self.fc1(cls[:, 1, :])))))
        if 2 in self.use_emb:
            emb.append(self.emb_dropout(self.bn2(self.silu(self.fc2(cls[:, 2, :])))))
        if 3 in self.use_emb:
            emb.append(self.emb_dropout(self.attn(x)))
        
        embedding = torch.cat(emb, dim=1)
        output = self.dropout(embedding)
        output = self.bn3(self.silu(self.fc3(output))).unsqueeze(1)
        
        if self.multiloss:
            embedding = embedding.reshape(B, len(self.use_emb)-1, H)
            return output, embedding
        return output
