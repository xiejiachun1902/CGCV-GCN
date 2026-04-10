import math
import cmath

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# import Complex_functions as CF

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
class C_softmax(nn.Module):
    def __init__(self,dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, z):
        exp_z = torch.exp(z)
        exp_sum = torch.sum(exp_z, dim=self.dim, keepdim=True)
        softmax_z = exp_z / exp_sum
        return softmax_z
class C_ReLU(torch.nn.Module):# 相位在（0，π/2）范围的复数保留，其他置零
    def __init__(self):
        super(C_ReLU, self).__init__()

    def forward(self, x):
        # 将复数从实部虚部表示转换为极坐标表示
        amplitude = torch.abs(x)  # 计算幅度
        phase = torch.angle(x)    # 计算相位

        # 判断相位是否在 0 到 π/2 范围内，并进行处理
        phase_in_range = (phase >= 0) & (phase <= 0.5 * torch.pi)
        phase_in_range = phase_in_range.float()

        amplitude = amplitude * phase_in_range  # 不满足条件的幅值置零
        phase = phase * phase_in_range          # 不满足条件的相位置零

        # 将处理后的幅度和相位重新转换为复数的实部和虚部
        real_part = amplitude * torch.cos(phase)
        imag_part = amplitude * torch.sin(phase)

        # 构造处理后的复数张量
        output = torch.complex(real_part, imag_part)

        return output
class C_Amplitude_Norm(nn.Module):
    def __init__(self):
        super(C_Amplitude_Norm, self).__init__()

    def normalize_amplitude_phase(self, z):
        # Separate real and imaginary parts
        real_part = z.real
        imag_part = z.imag

        # Compute amplitude and phase
        amplitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.atan2(imag_part, real_part)

        # Normalize amplitude to 1
        amplitude = amplitude / amplitude.max()

        # Normalize phase to (0, 2*pi)
        # phase = phase % (2 * torch.tensor([3.14159]))

        # Convert back to Cartesian form
        real_part = amplitude * torch.cos(phase)
        imag_part = amplitude * torch.sin(phase)

        # Create normalized complex tensor
        normalized_z = torch.complex(real_part, imag_part)

        return normalized_z

class C_sigmoid(nn.Module):
    def __init__(self,eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self,z):

        output = 1 / (1 + torch.exp(-(z + self.eps)))
        return output

class C_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias)
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.xavier_uniform_(self.bias)
    def init(self, mode, mean, std):
        # Initialize weight and bias with normal distribution
        if mode=='weight':
            nn.init.normal_(self.weight, mean=mean, std=std)
        if mode == 'bias':
            nn.init.normal_(self.bias, mean=mean, std=std)
    def forward(self, z):
        return torch.matmul(z, self.weight.T) + self.bias

# class C_Conv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=1, groups=1):
#         super().__init__()
#         assert in_channels % groups == 0, "In_channels should be an integer multiple of groups."
#         assert out_channels % groups == 0, "Out_channels should be an integer multiple of groups."
#         if isinstance(padding, int):
#             padding = (padding, padding)
#         if isinstance(kernel_size, int):
#             kernel_size = (kernel_size, kernel_size)
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.weight = nn.Parameter(torch.randn((out_channels, in_channels // groups, *kernel_size), dtype=torch.cfloat))
#         self.bias = nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat)) if bias else None
#     def branch_init(self, branches):
#         weight = self.weight
#         n = weight.size(0)
#         k1 = weight.size(1)
#         k2 = weight.size(2)
#         k3 = weight.size(3)
#         std = math.sqrt(2. / (n * k1 * k2 * k3 * branches))
#         nn.init.normal_(weight, 0, std)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, 0)
#     def init(self):
#         n = self.weight.size(0)  # 输出通道数
#         fan_out = n * self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
#         std = math.sqrt(2.0 / fan_out)# 标准差，nn.init.kaiming_normal_(conv.weight, mode='fan_out')
#         with torch.no_grad():
#             self.weight.normal_(0, std)
#             if self.bias is not None:
#                 self.bias.zero_()
#     def forward(self, z):
#         if not z.dtype == torch.cfloat:
#             z = torch.complex(z, torch.zeros_like(z))
#         out = torch.nn.functional.conv2d(input=z, weight=self.weight, bias=self.bias, stride=self.stride,
#                                          padding=self.padding, dilation=self.dilation, groups=self.groups)
#         return out
class C_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, dilation=1, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)
        # self.weight = self.conv2d.weight
        # self.bias = self.conv2d.bias
    def branch_init(self, branches):
        weight = self.conv2d.weight
        n = weight.size(0)
        k1 = weight.size(1)
        k2 = weight.size(2)
        nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
        if self.conv2d.bias is not None:
            nn.init.constant_(self.conv2d.bias, 0)
        # nn.init.constant_(self.conv2d.bias, 0)
    def init(self):
        n = self.conv2d.weight.size(0)  # 输出通道数
        fan_out = n * self.conv2d.weight.size(1) * self.conv2d.weight.size(2) * self.conv2d.weight.size(3)
        std = math.sqrt(2.0 / fan_out)
        with torch.no_grad():
            self.conv2d.weight.normal_(0, std)
            if self.conv2d.bias is not None:
                self.conv2d.bias.zero_()
    def forward(self, x):
        N, C, T, V = x.size()
        r = x.real  # data1的实部数据
        p = x.imag  # 虚部数据
        z_cat = torch.cat((r, p), dim=1)
        z_conv2d = self.conv2d(z_cat)
        r = z_conv2d[:, :self.out_channels, :, :]  # data1的实部数据
        p = z_conv2d[:, self.out_channels:, :, :]  # 虚部数据
        z_out = torch.complex(r, p)
        return z_out
class C_BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Initialize parameters if affine is True
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features, dtype=torch.cfloat))
            self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.cfloat))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Initialize running mean and variance
        self.running_mean = torch.zeros(num_features, dtype=torch.cfloat)
        self.running_var = torch.ones(num_features, dtype=torch.cfloat)

        # Initialize batch statistics tracking
        self.reset_running_stats()
    def reset_running_stats(self):
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long)
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
    def forward(self, x):
        if not x.dtype == torch.cfloat:
            x = torch.complex(x, torch.zeros_like(x))

        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            n = x.numel() / self.num_features
            # Update running mean and variance
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * n / (n - 1)
            # Normalize input using batch statistics
            normalized = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # Normalize input using running statistics
            normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        # Apply affine transformation if affine is True
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized
    def extra_repr(self):
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}'
class C_BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        # affine：代表gamma，beta是否可学。如果设为True，代表两个参数是通过学习得到的；如果设为False，代表两个参数是固定值，默认情况下，gamma是1，beta是0。
        # track_running_stats：BatchNorm2d中存储的的均值和方差是否需要更新，若为True，表示需要更新；反之不需要更新。更新公式参考momentum参数介绍
        super(C_BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
    def init(self, scale=1.0):
        if self.affine:
            nn.init.constant_(self.weight, scale)  # 设置 gamma 初始化为指定的 scale
            nn.init.constant_(self.bias, 0.0)      # 设置 beta 初始化为 0.0
    def forward(self, input):
        # 计算每个通道的均值和方差
        mean = input.mean(dim=(0, 2, 3), keepdim=False)
        var = input.var(dim=(0, 2, 3), unbiased=False, keepdim=False)

        if self.training:
            # 更新运行统计信息
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                self.num_batches_tracked += 1
        # 标准化输入数据
        normalized = (input - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        # 应用缩放和位移
        if self.affine:
            normalized = normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return normalized
# def conv_branch_init(conv, branches):
#     weight = conv.weight
#     n = weight.size(0)
#     k1 = weight.size(1)
#     k2 = weight.size(2)
#     nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
#     nn.init.constant_(conv.bias, 0)
class unit_Pluralization(nn.Module):# 复数化，并改变通道数
    def __init__(self, in_channels, out_channels, device='cuda'):
        super(unit_Pluralization, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        # self.norm_r = nn.LayerNorm(self.in_channels)
        # self.norm_p = nn.LayerNorm(self.in_channels)
        # self.norm = nn.LayerNorm(self.in_channels)
        self.line_r = nn.Linear(in_features = self.in_channels, out_features = self.out_channels)
        self.line_p = nn.Linear(in_features = self.in_channels, out_features = self.out_channels)

        self.line_phase = nn.Linear(in_features = self.in_channels, out_features = self.out_channels)

        self.cline = C_Linear(in_features=self.in_channels, out_features=self.out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.to(self.device)
        # # x 的初始形状为 (N, C, T, V)，其中 C=3
        # # 调整形状为 (N, T, V, C)，以适应线性层
        # x = x.permute(0, 2, 3, 1).contiguous()
        # # 展平前三个维度为一个维度，以适应线性层
        # x = x.reshape(N * T * V, C)
        # r = torch.relu(self.line_r(x))
        # p = torch.relu(self.line_p(x))
        # # 将x恢复到四维形状，此时特征维变为16
        # r = r.reshape(N, T, V, self.out_channels).permute(0, 3, 1, 2).contiguous()
        # p = p.reshape(N, T, V, self.out_channels).permute(0, 3, 1, 2).contiguous()
        # complex_x = torch.complex(r, p)


        # x 的初始形状为 (N, C, T, V)，其中 C=3
        # 调整形状为 (N, T, V, C)，以适应线性层
        x = x.permute(0, 2, 3, 1).contiguous()
        # 展平前三个维度为一个维度，以适应线性层
        x = x.reshape(N * T * V, C)
        phase = torch.relu(self.line_phase(x))

        # 将x恢复到四维形状，此时特征维变为16

        phase = phase.reshape(N, T, V, self.out_channels).permute(0, 3, 1, 2).contiguous()
        amplitude = torch.ones_like(phase)

        real_part = amplitude * torch.cos(phase)
        imag_part = amplitude * torch.sin(phase)
        complex_x = torch.complex(real_part, imag_part)
        return complex_x
class unit_ReversePluralization(nn.Module):# 复数化，并改变通道数
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.line = nn.Linear(in_features * 2, out_features)
        self.conv = nn.Conv2d(in_channels=in_features * 2, out_channels=out_features,kernel_size=1,stride=1,padding=0)

        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias)

        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.xavier_uniform_(self.bias)
    def init(self, mode, mean, std):
        # Initialize weight and bias with normal distribution
        if mode=='weight':
            nn.init.normal_(self.weight, mean=mean, std=std)
        if mode == 'bias':
            nn.init.normal_(self.bias, mean=mean, std=std)
    def forward(self, z):
        # if not z.dtype == torch.cfloat:
        #     z = torch.complex(z, torch.zeros_like(z))

        # N, C, T, V = z.shape
        # r = z.real
        # p = z.imag
        # x = torch.cat((r, p), dim=1)
        # x = self.conv(x)


        N, C, T, V = z.shape
        r = z.real
        p = z.imag
        # 计算模长和相位
        modulus = torch.abs(z)  # 模长
        phase = torch.angle(z)  # 相位
        x = phase

        # print(torch.isinf(phase)
        # print(torch.isnan(phase)

        return x
class real_tcn(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0, p=0.2,residual=True):
        super(real_tcn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=0,
                              stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(p=p)

        self.res = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn_res = nn.BatchNorm1d(out_channels)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        elif in_channels * 2 == out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        else:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # print(x.shape)
        res = self.residual(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.drop(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + res

        return x


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.cconv = C_Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.cbn = C_BatchNorm2d(out_channels)
        self.crelu = C_ReLU()
        self.cconv.init()

        # bn_init(self.bn, 1)
        # conv_init(self.conv)
        self.cbn.init()

    def forward(self, x):
        x = self.cbn(self.cconv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(C_Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(C_Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(C_Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                C_Conv2d(in_channels, out_channels, 1),
                C_BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.cbn = C_BatchNorm2d(out_channels)
        self.csoft = C_softmax(dim = 1)
        # self.soft = nn.Softmax(-2)
        self.crelu = C_ReLU()

        for m in self.modules():
            if isinstance(m, C_Conv2d):
                m.init()
                # conv_init(m)
            elif isinstance(m, C_BatchNorm2d):
                m.init(1)
                # bn_init(m, 1)
        # bn_init(self.bn, 1e-6)
        self.cbn.init(1e-6)

        for i in range(self.num_subset):
            self.conv_d[i].branch_init(self.num_subset)
            # conv_branch_init(self.conv_d[i], self.num_subset)

            # self.conv_d[i].p_init(self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            # print(x.shape)
            # print(self.conv_a[i](x).shape)
            # print(self.inter_c)
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.csoft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.cbn(y)
        y += self.down(x)
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.crelu = C_ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return x



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveAvgPool2d(1)  # nn.AdaptiveMaxPool2d(1)

        self.cfc1 = C_Conv2d(in_planes, in_planes, 1, bias=False)
        self.crelu1 = C_ReLU()
        self.cfc2 = C_Conv2d(in_planes, in_planes, 1, bias=False)

        self.csigmoid = C_sigmoid()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.cfc2(self.crelu1(self.cfc1(self.max_pool(x))))
        out = max_out
        return self.csigmoid(out)


class fusion(nn.Module):
    def __init__(self, T_in_channels):
        super(fusion, self).__init__()
        self.att_T_p = ChannelAttention(T_in_channels)
        self.att_N_p = ChannelAttention(16)
        self.att_T_m = ChannelAttention(T_in_channels)
        self.att_N_m = ChannelAttention(16)

    def forward(self, x_p, x_k):
        # B*C*T*N 自适应融合，T和N各自轴上
        # 情感特征的参数化，或者平均方式的参数化。
        B, C, T, N = x_p.size()
        x_p_T = x_p.permute(0, 2, 1, 3)
        x_p_N = x_p.permute(0, 3, 2, 1)
        x_k_T = x_k.permute(0, 2, 1, 3)
        x_k_N = x_k.permute(0, 3, 2, 1)


        att_N_p_map = (self.att_N_p(x_p_N)).permute(0, 3, 2, 1)
        x_p_mid = (x_p * att_N_p_map).permute(0, 2, 1, 3)
        att_T_p_map = (self.att_T_p(x_p_mid)).permute(0, 2, 1, 3)

        att_N_m_map = (self.att_N_m(x_k_N)).permute(0, 3, 2, 1)
        x_k_mid = (x_k * att_N_m_map).permute(0, 2, 1, 3)
        att_T_m_map = (self.att_T_m(x_k_mid)).permute(0, 2, 1, 3)

        x_p = x_p + x_k * att_T_m_map
        x_k = x_k + x_p * att_T_p_map

        # x_p=x_p+x_k*att_T_m_map*att_N_m_map
        # x_k=x_k+x_p*att_T_p_map*att_N_p_map
        # x_p = x_p + x_k * (att_T_m_map+att_N_m_map)
        # x_k = x_k + x_p * (att_T_p_map+att_T_p_map)

        return x_p, x_k

class fusion_block(nn.Module):
    def __init__(self,channels_1 ,channels_2, channels_f,channels_x_low, r_1, fusion_dropout_rate):
        super(fusion_block, self).__init__()

        self.c_1 = channels_1#复数层
        self.c_2 = channels_2#实数层
        self.c_fuse = 2*channels_1 + channels_2 + channels_f
        self.c_fuse_out = channels_1
        self.dr_rate = fusion_dropout_rate
        # LFPB
        self.LFPB_maxpool = nn.AdaptiveMaxPool2d(1)
        self.LFPB_avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channels_1*2, channels_1*2 // r_1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels_1*2 // r_1, channels_1*2, 1, bias=False)
        )
        self.LFPB_sigmoid = nn.Sigmoid()
        self.LFPB_conv = nn.Conv2d(in_channels=2*channels_2, out_channels=channels_2, kernel_size=(1,1))

        # GFPB
        self.GFPB_maxpool = nn.MaxPool1d(kernel_size=channels_2)
        self.GFPB_avgpool = nn.AvgPool1d(kernel_size=channels_2)
        self.GFPB_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.GFPB_sigmoid = nn.Sigmoid()
        # cross_fusion
        self.conv_gate_1 = nn.Conv1d(in_channels=channels_2, out_channels=channels_2, kernel_size=1)
        self.conv_gate_2 = nn.Conv1d(in_channels=channels_2, out_channels=channels_2, kernel_size=1)
        self.gated_relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # conv_fusion
        self.f_norm = LayerNorm(self.c_2, eps=1e-6, data_format="channels_first")
        self.f_conv1 = nn.Conv1d(in_channels=self.c_2, out_channels=self.c_2, kernel_size=3, padding=1)
        self.f_gelu = nn.GELU()
        self.f_bn1 = nn.BatchNorm1d(self.c_2)
        self.f_conv2 = nn.Conv1d(in_channels=self.c_2, out_channels=4*self.c_2, kernel_size=1)
        self.f_conv3 = nn.Conv1d(in_channels=4*self.c_2, out_channels=self.c_fuse_out, kernel_size=1)
        self.f_bn2 = nn.BatchNorm1d(self.c_fuse_out)
        # forward
        if self.dr_rate > 0:
            self.f_dr = nn.Dropout(p=self.dr_rate)
        else:
            self.f_dr = nn.Identity()
        if channels_x_low > 0:
            self.fp_conv = nn.Conv1d(in_channels=channels_x_low, out_channels=channels_f, kernel_size=1, stride=2)
        self.conv = nn.Conv2d(in_channels=2*channels_1,out_channels=channels_1,kernel_size=1,stride=1,padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 16))
    def LFPB(self, x):
        max_x = self.LFPB_maxpool(x)  # N,2C,T,V
        avg_x = self.LFPB_avgpool(x)
        max_out = self.se(max_x)  # N,2C,T,V
        avg_out = self.se(avg_x)
        channel_att_out = self.LFPB_sigmoid(max_out + avg_out) * x
        out_2 = channel_att_out + x
        out = self.LFPB_conv(out_2)  # N,C,T,V
        return out
    def GFPB(self, x):
        N, C, T= x.shape
        x_mid = x.permute(0, 2, 1)  # N, T, C
        max_x = self.GFPB_maxpool(x_mid)  # 得到N, T, 1
        avg_x = self.GFPB_avgpool(x_mid)
        cat_x = torch.cat([max_x, avg_x],dim=2)  # 得到N, T, 2
        cat_x = cat_x.permute(0, 2, 1)  # N, 2, T
        x_out = self.GFPB_conv(cat_x)
        out = self.GFPB_sigmoid(x_out) * x
        return out
    def conv_fusion(self,fuse):
        fusion_res = fuse  # 残差
        fuse = self.f_norm(fuse)
        fuse = self.f_conv1(fuse)  # N,C,T
        fuse = self.f_gelu(fuse)
        fuse += fusion_res
        fuse = self.f_bn1(fuse)
        fuse = self.f_conv2(fuse)  # N,4C,T
        fuse = self.f_gelu(fuse)
        fuse = self.f_conv3(fuse)  # N,C,T
        fuse = self.f_bn2(fuse)
        return fuse
    def cross_fusion(self,LFPB,GFPB,x_low):
        gate_1 = self.gated_relu(self.conv_gate_1(GFPB))
        LFPB1 = gate_1 * LFPB
        gate_2 = self.gated_relu(self.conv_gate_2(GFPB))
        cf_mid = gate_2 * x_low
        fusion_1 = self.tanh(GFPB + LFPB1)
        cf_out = cf_mid + fusion_1
        return cf_out
    def forward(self, data1, data2, x_low):
        N, C, T, V = data1.shape
        d1_r = data1.real
        d1_i = data1.imag
        d1 = torch.cat([d1_r, d1_i], dim=1)  # N, 2C, T, V
        LFPB = self.LFPB(d1)  # N, C, T, V
        LFPB = self.avg_pool(LFPB)
        LFPB = LFPB.squeeze(dim=3)  # N, C, T
        GFPB = self.GFPB(data2)# N,C,T
        if x_low is not None:
            x_low = self.fp_conv(x_low)
            res = x_low
        else:
            res = 0
            x_low = torch.ones(N, C, T, dtype=data2.dtype, device=data2.device)
        fuse = self.cross_fusion(LFPB, GFPB, x_low)
        fuse = self.conv_fusion(fuse)
        fuse = self.f_dr(fuse)
        f_out = fuse + res
        return f_out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            x = x.permute(0, 2, 1)
            out = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            out = out.permute(0, 2, 1)
            return out

class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_constraints=31, graph=None, graph_args=dict(), in_channels_p=3,
                 in_channels_k=8,in_channels_a=31,pk_dropout_rate=0,a_dropout_rate=0,fusion_dropout_rate=0):
        super(Model, self).__init__()
        self.p_dr = pk_dropout_rate
        self.k_dr = pk_dropout_rate
        self.a_dr = a_dropout_rate
        self.f_dr = fusion_dropout_rate

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_k = nn.BatchNorm1d(in_channels_k * num_point)
        # self.data_bn_a = nn.BatchNorm1d(in_channels_k * num_point)

        self.P_p = unit_Pluralization(in_channels_p, in_channels_p)
        self.P_k = unit_Pluralization(in_channels_k, in_channels_k)
        # self.P_a = unit_Pluralization(in_channels_a, in_channels_a)
        self.RP_p = unit_ReversePluralization(in_features = 256, out_features = 256)
        self.RP_k = unit_ReversePluralization(in_features = 256, out_features = 256)

        self.l1_p = TCN_GCN_unit(in_channels_p, 64, A, residual=False)
        self.l1_k = TCN_GCN_unit(in_channels_k, 64, A, residual=False)

        self.l2_p = TCN_GCN_unit(64, 64, A)
        self.l3_p = TCN_GCN_unit(64, 64, A)
        self.l4_p = TCN_GCN_unit(64, 64, A)
        self.l5_p = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_p = TCN_GCN_unit(128, 128, A)
        self.l7_p = TCN_GCN_unit(128, 128, A)
        self.l8_p = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_p = TCN_GCN_unit(256, 256, A)
        self.l10_p = TCN_GCN_unit(256, 256, A)

        self.l2_k = TCN_GCN_unit(64, 64, A)
        self.l3_k = TCN_GCN_unit(64, 64, A)
        self.l4_k = TCN_GCN_unit(64, 64, A)
        self.l5_k = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_k = TCN_GCN_unit(128, 128, A)
        self.l7_k = TCN_GCN_unit(128, 128, A)
        self.l8_k = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_k = TCN_GCN_unit(256, 256, A)
        self.l10_k = TCN_GCN_unit(256, 256, A)

        self.pk_fusion1 = fusion(48)
        self.pk_fusion2 = fusion(24)
        self.pk_fusion3 = fusion(12)

        """ 情感流
                   """
        self.data_bn_a = nn.BatchNorm1d(in_channels_a)

        self.ll1_a = nn.Conv1d(in_channels=in_channels_a, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.ll2_a = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.ll3_a = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.ll4_a = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)


        self.l1_a = real_tcn(in_channels=in_channels_a, out_channels=64, kernel_size=1, stride=1, padding=0, p=0)
        self.l2_a = real_tcn(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, p=0)
        self.l3_a = real_tcn(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, p=0)
        self.l4_a = real_tcn(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, p=0)

        self.l5_a = real_tcn(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, p=0)
        self.l6_a = real_tcn(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, p=0)

        self.fc1_classifier_a = nn.Linear(256,num_class)

        self.pa_fusion_1 = fusion_block(64, 64, 64, 1, 8, self.f_dr)
        self.pa_fusion_2 = fusion_block(128, 128, 128, 64, 8, self.f_dr)
        self.pa_fusion_3 = fusion_block(256, 256, 256, 128, 8, self.f_dr)

        self.ka_fusion_1 = fusion_block(64, 64, 64, 1, 8, self.f_dr)
        self.ka_fusion_2 = fusion_block(128, 128, 128, 64, 8, self.f_dr)
        self.ka_fusion_3 = fusion_block(256, 256, 256, 128, 8, self.f_dr)

        self.classifier_fusion_pa = nn.Linear(256, num_class)
        self.classifier_fusion_ka = nn.Linear(256, num_class)

        self.fc1_classifier_p = nn.Linear(256, num_class)
        self.fc1_classifier_k = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, num_constraints * 48)

        nn.init.normal_(self.fc1_classifier_k.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (num_constraints * 48)))

        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_k, 1)

    def forward(self, x_p, x_k, x_a):
        N, C_p, T, V, M = x_p.size()
        N, C_k, T, V, M = x_k.size()
        # 调整并做bn
        x_p = x_p.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_p, T)
        x_k = x_k.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_k, T)
        x_p = self.data_bn_p(x_p)
        x_k = self.data_bn_k(x_k)
        x_p = x_p.view(N, M, V, C_p, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_p, T, V)
        x_k = x_k.view(N, M, V, C_k, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_k, T, V)
        # 复数化
        x_p = self.P_p(x_p)
        x_k = self.P_k(x_k)

        # 情感流af
        x_a = x_a.reshape(N, T, -1)
        N, T, C_a = x_a.size()
        x_a = x_a.permute(0, 2, 1)# N,C,T

        x_a = self.data_bn_a(x_a)

        fuse_pa = None
        fuse_ka = None


        x_p = self.l1_p(x_p)
        x_k = self.l1_k(x_k)
        x_a = self.l1_a(x_a)


        x_p = self.l2_p(x_p)
        x_p = self.l3_p(x_p)
        x_p = self.l4_p(x_p)
        x_k = self.l2_k(x_k)
        x_k = self.l3_k(x_k)
        x_k = self.l4_k(x_k)
        x_a = self.l2_a(x_a)

        fuse_pa = self.pa_fusion_1(x_p, x_a, fuse_pa)
        fuse_ka = self.ka_fusion_1(x_k, x_a, fuse_ka)
        x_p, x_k = self.pk_fusion1(x_p, x_k)


        x_p = self.l5_p(x_p)
        x_p = self.l6_p(x_p)
        x_p = self.l7_p(x_p)

        x_k = self.l5_k(x_k)
        x_k = self.l6_k(x_k)
        x_k = self.l7_k(x_k)

        x_a = self.l3_a(x_a)


        fuse_pa = self.pa_fusion_2(x_p, x_a, fuse_pa)
        fuse_ka = self.ka_fusion_2(x_k, x_a, fuse_ka)
        x_p, x_k = self.pk_fusion2(x_p, x_k)

        x_p = self.l8_p(x_p)
        x_p = self.l9_p(x_p)
        x_p = self.l10_p(x_p)

        x_k = self.l8_k(x_k)
        x_k = self.l9_k(x_k)
        x_k = self.l10_k(x_k)

        x_a = self.l4_a(x_a)

        fuse_pa = self.pa_fusion_3(x_p, x_a, fuse_pa)
        fuse_ka = self.ka_fusion_3(x_k, x_a, fuse_ka)
        x_p, x_k = self.pk_fusion3(x_p, x_k)


        x_p = self.RP_p(x_p)
        x_k = self.RP_k(x_k)
        # N*M,C,T,V
        c_new_k = x_k.size(1)
        x_k = x_k.view(N, M, c_new_k, -1)
        x_k = x_k.mean(3).mean(1)

        c_new_p = x_p.size(1)
        x_p = x_p.view(N, M, c_new_p, -1)
        x_p = x_p.mean(3).mean(1)

        c_new_a = x_a.size(1)
        x_a = x_a.view(N, M, c_new_a, -1)
        x_a = x_a.mean(3).mean(1)

        c_new_pa = fuse_pa.size(1)
        fuse_pa = fuse_pa.reshape(N, M, c_new_pa, -1)
        fuse_pa = fuse_pa.mean(3).mean(1)

        c_new_am = fuse_ka.size(1)
        fuse_ka = fuse_ka.reshape(N, M, c_new_am, -1)
        fuse_ka = fuse_ka.mean(3).mean(1)

        return self.fc1_classifier_p(x_p), self.fc1_classifier_k(x_k), self.fc1_classifier_a(x_a), self.classifier_fusion_pa(fuse_pa), self.classifier_fusion_ka(fuse_ka)



