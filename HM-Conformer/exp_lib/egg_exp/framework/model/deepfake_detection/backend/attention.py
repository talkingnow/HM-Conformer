import torch
import torch.nn as nn

"""
Code from https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master
"""

class Attention(nn.Module):
    def __init__(self, feature_processing_name, in_dim, hidden_dim, num_head=1, out_dim=2, input_mean_std=True):
        super(Attention, self).__init__()
        self.input_mean_std = input_mean_std
        
        if feature_processing_name == 'LCNN':
            in_dim = (in_dim // 16) * 32
        elif feature_processing_name == 'ECAPA_TDNN':
            # CAUTION 
            # There is no BatchNorm1d before Linear layer
            in_dim = (in_dim * 3) // 2
        
        # input mean and std to attention pooling
        if self.input_mean_std:
            in_dim = in_dim * 3
            
        # add noise when extracting std
        self.m_pooling = SelfWeightedPooling(in_dim, num_head=num_head, mean_only=False)
        
        # mean, std
        self.m_output_act = nn.Linear(in_dim* 2, hidden_dim)
        self.m_bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        # (batchsize, length, feature_dim)
        assert len(x.size()) == 3, f'Input size error in pooling. Need 3, but get {len(x.size())}'
        if self.input_mean_std:
            time = x.size(1)
            temp1 = torch.mean(x, dim=1, keepdim=True).repeat(1, time, 1)
            temp2 = torch.sqrt(torch.var(x, dim=1, keepdim=True).clamp(min=1e-9)).repeat(1, time, 1)
            x = torch.cat((x, temp1, temp2), dim=2)
            
        mean_std = self.m_pooling(x)
        output = self.m_output_act(mean_std)
        
        return output

class SelfWeightedPooling(nn.Module):
    def __init__(self, feature_dim, num_head=1, mean_only=False):
        super(SelfWeightedPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mean_only = mean_only
        self.noise_std = 1e-5
        self.num_head = num_head

        # transformation matrix (num_head, feature_dim)
        self.mm_weights = nn.Parameter(
            torch.Tensor(num_head, feature_dim), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.mm_weights)
        
    def forward(self, inputs, get_w=False, tanh=True):
        # batch size
        batch_size = inputs.size(0)
        # feature dimension
        feat_dim = inputs.size(2)
        
        # input is (batch, legth, feature_dim)
        # change mm_weights to (batchsize, feature_dim, num_head)
        # weights will be in shape (batchsize, length, num_head)
        weights = torch.bmm(inputs, 
                            self.mm_weights.permute(1, 0).contiguous()\
                            .unsqueeze(0).repeat(batch_size, 1, 1))
        
        # attention (batchsize, length, num_head)
        if tanh:
            attentions = nn.functional.softmax(torch.tanh(weights), dim=1)    
        else: 
            attentions = nn.functional.softmax(weights, dim=1)  
        
        # apply attention weight to input vectors
        if self.num_head == 1:
            # We can use the mode below to compute self.num_head too
            # But there is numerical difference.
            #  original implementation in github
            
            # elmentwise multiplication
            # weighted input vector: (batchsize, length, feature_dim)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            # weights_mat = (batch * length, feat_dim, num_head)
            weighted = torch.bmm(
                inputs.view(-1, feat_dim, 1), 
                attentions.view(-1, 1, self.num_head))
            
            # weights_mat = (batch, length, feat_dim * num_head)
            weighted = weighted.view(batch_size, -1, feat_dim * self.num_head)
            
        # pooling
        if self.mean_only:
            # only output the mean vector
            representations = weighted.sum(1)
        else:
            # output the mean and std vector
            noise = self.noise_std * torch.randn(
                weighted.size(), dtype=weighted.dtype, device=weighted.device)

            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)
            # concatenate mean and std
            representations = torch.cat((avg_repr,std_repr),1)

        # done
        if get_w:
            return representations, attentions.squeeze(-1)
        return representations