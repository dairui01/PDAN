import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import copy


class PDAN(nn.Module):
    def __init__(self, num_stages=1, num_layers=5, num_f_maps=512, dim=1024, num_classes=157):
        super(PDAN, self).__init__()
        self.stage1 = SSPDAN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SSPDAN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        #doesn do anything
        for s in self.stages:
            out = s(out * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

class SSPDAN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SSPDAN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(PDAN_Block(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        #TODO: change to linear, but it should work worse
        #self.conv_out = nn.Linear(num_f_maps, num_classes)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        #out = nn.Transformer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class PDAN_Block(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(PDAN_Block, self).__init__()
        self.conv_attention=DAL(in_channels, out_channels, kernel_size=3, padding=dilation, dilated=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_attention(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        #return out * mask[:, 0:1, :]
        return (x + out) * mask[:, 0:1, :]

class DAL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilated=1, groups=1, bias=False):
        super(DAL, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilated = dilated
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        self.rel_t = nn.Parameter(torch.randn(out_channels, 1, kernel_size), requires_grad=True)
        #self.linear = nn.Linear(in_channels, out_channels)
        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        #self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.reset_parameters()


    def forward(self, x):
        batch, channels, time = x.size()
        padded_x = F.pad(x, (self.padding, self.padding))
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        # v_out = self.value_conv(padded_x)
        kernal_size = 2*self.dilated + 1
        #print(k_out.shape)
        k_out = k_out.unfold(2, kernal_size, self.stride)  # unfold(dim, size, step)
        #print(k_out.shape)
        k_out=torch.cat((k_out[:,:,:,0].unsqueeze(3),k_out[:,:,:,0+self.dilated].unsqueeze(3),k_out[:,:,:,0+2*self.dilated].unsqueeze(3)),dim=3)  #dilated
        #print(k_out.shape,'\n\n')
        # v_out = v_out.unfold(2, kernal_size, self.stride)
        # v_out=torch.cat((v_out[:,:,:,0].unsqueeze(3),v_out[:,:,:,0+self.dilated].unsqueeze(3),v_out[:,:,:,0+2*self.dilated].unsqueeze(3)),dim=3)  #dilated
        # print(self.rel_t.shape)
        # print(k_out.shape)
        #TODO: I could do this afterwards, but have to be careful with dimensions. Just moving it lower decreases performance and increases overfitting
        v_out = k_out + self.rel_t
        #v_out = self.linear(k_out)
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, time, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, time, -1)
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, time, 1)
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnctk,bnctk -> bnct', out, v_out).view(batch, -1, time)
        return out

    def reset_parameters(self):
        init.kaiming_normal(self.key_conv.weight, mode='fan_out')
        #init.kaiming_normal(self.value_conv.weight, mode='fan_out')
        init.kaiming_normal(self.query_conv.weight, mode='fan_out')
        #init.kaiming_normal(self.linear.weight, mode='fan_out')
        init.normal(self.rel_t, 0, 1)


