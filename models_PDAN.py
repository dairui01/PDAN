import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


class DAL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(DAL, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilated = dilation

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.reset_parameters()
        # self.unfold=torch.nn.Unfold(kernel_size=3, dilation=1, padding=0, stride=1)

    def forward(self, x):
        batch, channels, time = x.size()

        padded_x = F.pad(x, (self.padding, self.padding))

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        kernal_size = 2 * self.dilated + 1
        k_out = k_out.unfold(2, kernal_size, self.stride)  # unfold(dim, size, step)
        k_out = torch.cat((k_out[:, :, :, 0].unsqueeze(3), k_out[:, :, :, 0 + self.dilated].unsqueeze(3),
                           k_out[:, :, :, 0 + 2 * self.dilated].unsqueeze(3)), dim=3)  # dilated
        v_out = v_out.unfold(2, kernal_size, self.stride)
        v_out = torch.cat((v_out[:, :, :, 0].unsqueeze(3), v_out[:, :, :, 0 + self.dilated].unsqueeze(3),
                           v_out[:, :, :, 0 + 2 * self.dilated].unsqueeze(3)), dim=3)  # dilated
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, time, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, time, -1)
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, time, 1)
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnctk,bnctk -> bnct', out, v_out).view(batch, -1, time)
        return out

    def reset_parameters(self):
        init.kaiming_normal(self.key_conv.weight, mode='fan_out')
        init.kaiming_normal(self.value_conv.weight, mode='fan_out')
        init.kaiming_normal(self.query_conv.weight, mode='fan_out')


class PDAN(nn.Module):
    def __init__(self, inter_dim=256, input_dim=1024, num_classes=157):
        super(PDAN, self).__init__()
        self.inter_dim = int(inter_dim)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.conv1 = DAL(self.inter_dim, self.inter_dim, 3, padding=1, dilation=1)
        self.conv2 = DAL(self.inter_dim, self.inter_dim, 3, padding=2, dilation=2)
        self.conv3 = DAL(self.inter_dim, self.inter_dim, 3, padding=4, dilation=4)
        self.conv4 = DAL(self.inter_dim, self.inter_dim, 3, padding=8, dilation=8)
        self.conv5 = DAL(self.inter_dim, self.inter_dim, 3, padding=16, dilation=16)

        self.Drop1 = nn.Dropout()
        self.Drop2 = nn.Dropout()
        self.Drop3 = nn.Dropout()
        self.Drop4 = nn.Dropout()
        self.Drop5 = nn.Dropout()

        self.bottle0 = nn.Conv1d(self.input_dim, self.inter_dim, 1)
        self.bottle1 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle2 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle3 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle4 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle5 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle6 = nn.Conv1d(self.inter_dim, self.num_classes, 1)

    def forward(self, x, mask):
        out0 = self.bottle0(x)

        out1 = F.relu(self.conv1(out0))
        out1 = self.bottle1(out1)
        out1 = self.Drop1(out1)
        out1 = (out0 + out1) * mask[:, 0:1, :]

        out2 = F.relu(self.conv2(out1))
        out2 = self.bottle2(out2)
        out2 = self.Drop2(out2)
        out2 = (out1 + out2) * mask[:, 0:1, :]

        out3 = F.relu(self.conv3(out2))
        out3 = self.bottle3(out3)
        out3 = self.Drop3(out3)
        out3 = (out2 + out3) * mask[:, 0:1, :]

        out4 = F.relu(self.conv4(out3))
        out4 = self.bottle4(out4)
        out4 = self.Drop4(out4)
        out4 = (out3 + out4) * mask[:, 0:1, :]

        out5 = F.relu(self.conv5(out4))
        out5 = self.bottle5(out5)
        out5 = self.Drop5(out5)
        out5 = (out4 + out5) * mask[:, 0:1, :]

        out6 = self.bottle6(out5) * mask[:, 0:1, :]

        return out6



