import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # (64,32,170,12) (170,170)
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        # 2 * 3 + 1 = 7 * 32 = 224
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)  # 224,32
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # x [64, 32, 170, 12] (出度,入度,自加)
        out = [x]
        for a in support:
            # print(a.shape)
            # exit(0)

            # torch.Size([207, 207])
            # print(x.shape)
            # exit(0)
            x1 = self.nconv(x, a)  # [64,32,207,12]

            out.append(x1)
            # order = 2
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)  # [64,32,207,12]
                out.append(x2)
                x1 = x2
        # for _ in out:
        #     print(_.shape)
        # exit(0)
        h = torch.cat(out, dim=1)  # [64,224,207,12]
        # print(h.shape)
        h = self.mlp(h)  # [64,32,207,12]
        # print(h.shape)
        # exit(0)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # Conv2d(2,32,(1,1))
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0  # 2
        if supports is not None:
            self.supports_len += len(supports)

        # gcn_bool = True ; addaptadj = True aptinit = None
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        # blocks = 4
        for b in range(blocks):
            # additional_scope = 2 - 1 = 1
            additional_scope = kernel_size - 1
            new_dilation = 1
            # layers = 2
            for i in range(layers):
                # dilated convolutions
                # (32,32,(1,2),dilation=1)
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # (32,32,(1,2),1)
                # self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                #                                  out_channels=dilation_channels,
                #                                  kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                # self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                #                                      out_channels=residual_channels,
                #                                      kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                # self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                #                                  out_channels=skip_channels,
                #                                  kernel_size=(1, 1)))

                # (32,256,(1,1))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        # (256,512)
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        # # (512,256)
        # self.end_conv_new2 = nn.Conv2d(in_channels=end_channels, out_channels=skip_channels, kernel_size=(1, 1),
        #                                bias=True)
        #
        # # (256,128)
        # self.end_conv_new3 = nn.Conv2d(in_channels=skip_channels, out_channels=int(skip_channels / 2),
        #                                kernel_size=(1, 1), bias=True)
        #
        # # (128,64)
        # self.end_conv_new4 = nn.Conv2d(in_channels=int(skip_channels / 2), out_channels=int(skip_channels / 4),
        #                                kernel_size=(1, 1), bias=True)
        #
        # self.end_conv_new5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), bias=True)
        #
        # self.end_conv_new6 = nn.Conv2d(in_channels=32, out_channels=out_dim, kernel_size=(1, 1), bias=True)

        # (512,2)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        # input = [64, 1, 207, 13] 第一个为0
        in_len = input.size(3)  # 13
        # 这段代码不执行
        # 13 == 13
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input  # [64, 1, 207, 13]

        # Conv2d(2,32,(1,1))
        x = self.start_conv(x)  # torch.Size([64, 32, 207, 13])
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None

        # gcn_bool = True addaptadj = True supports!=None
        # Equation 5
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # 新建的连接关系
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        # blocks = 4, layers = 2
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # residual(上一层x)# -> dilate -|----|      * --(x!!!)------   gconv -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # skip# --------------------------------------->(s)+ ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)

            # torch.Size([64, 32, 207, 13]) => 1
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)  # [64,32,207,12]
            # print(filter.shape)
            filter = torch.tanh(filter)  # 激活函数
            gate = self.gate_convs[i](residual)  # [64,32,207,12]
            # print(gate.shape)
            # exit(0)
            gate = torch.sigmoid(gate)
            # 对应元素相乘
            x = filter * gate  # [64,32,207,12]
            # print(x)
            # print(x.shape)
            # exit(0)

            # parametrized skip connection

            # [64,32,207,12]
            s = x
            s = self.skip_convs[i](s)
            # [64, 256, 207, 12]
            # s.size(3)=>12
            # print(skip)
            # exit(0)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            # skip = s = [64,256,207,12]
            skip = s + skip  # 汇聚八层
            # print(skip.shape)
            # exit(0)

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # [64,32,207,12]
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            # [64,32,207,12]
            x = x + residual[:, :, :, -x.size(3):]
            # [64,32,207,12]
            x = self.bn[i](x)
            # print(x.shape)
            # exit(0)
        # exit(0)
        x = F.relu(skip)  # [64,256,170,1]
        # x = skip
        x = F.relu(self.end_conv_1(x))  # [64,512,170,1]
        x = self.end_conv_2(x)  # [64,12,170,1]
        #
        # # x = F.relu(x)
        # x = F.relu(self.end_conv_new2(x))
        # x = F.relu(self.end_conv_new3(x))
        # x = F.relu(self.end_conv_new4(x))
        # x = F.relu(self.end_conv_new5(x))
        # x = self.end_conv_new6(x)

        return x
