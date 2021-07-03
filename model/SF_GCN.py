import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gcn_helper



class STGCN_Fuse(nn.Module):
    """
    Covolution Fusion
    """
    def __init__(self, dim_in, A):
        super().__init__()
        self.fuse_f2s = nn.Conv2d(
                            dim_in,
                            dim_in * A.size(1),
                            kernel_size=(7, 3),
                            stride=(8, 1),
                            padding = (0, 1),
                            bias=False,
                            )
        self.bn = nn.BatchNorm2d(dim_in * A.size(1))
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x, A):
        x_s = x[0]
        x_f = x[1]
        fuse_f2s = self.fuse_f2s(x_f)
        fuse_f2s = self.bn(fuse_f2s)
        fuse_f2s = self.relu(fuse_f2s)
        x_s_fuse = torch.cat([x_s, fuse_f2s], 1)
        output = [x_s_fuse, x_f]
        return output

class SFGCN_Fuse(nn.Module):
    """
    Residual Fusion
    """
    def __init__(self, dim_in, A):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=1,
                stride=(1,1)
            ),
            nn.BatchNorm2d(dim_in)
        )
        self.relu = nn.ReLU(inplace=True)
        self.fuse_f2s = nn.Sequential(
        nn.Conv2d(
            dim_in,
            dim_in * A.size(1),
            kernel_size=(7, 3),
            stride=(8, 1),
            padding = (0, 1),
            bias=False,
                        ),
        nn.BatchNorm2d(dim_in * A.size(1)),
        nn.ReLU(inplace=True)
        )

    def forward(self, x, A):
        x_s = x[0]
        x_f = x[1]
        res_xf = self.residual(x_f)
        res_xf = self.relu(res_xf + x_f)
        fuse_f2s = self.fuse_f2s(res_xf)
        x_s_fuse = torch.cat([x_s, fuse_f2s], 1)
        output = [x_s_fuse, res_xf]
        return output

class SF_STGCN(nn.Module):
    def __init__(
                self,
                num_pathways,
                in_channels,
                max_frame,
                num_class,
                graph_args,
                beta_inv=8,
                edge_important_weight=True
                ):
        super().__init__()
        self.num_pathways = num_pathways

        # load graph
        self.graph = gcn_helper.Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.st_gcn_networks = nn.ModuleList((
        gcn_helper.sf_gcn(
                        in_channels=[
                                    in_channels, in_channels
                                        ],
                        out_channels=[
                                    64, 64 // beta_inv
                                    ],
                        kernel_size=kernel_size
                            ),
        STGCN_Fuse(
                    dim_in=64 // beta_inv ,
                    A=A,
                    ),
        gcn_helper.sf_gcn(
                        in_channels=[
                            64 + 64 // beta_inv * A.size(1), 
                            64 // beta_inv
                            ],
                        out_channels=[
                            128, 128 // beta_inv
                            ],
                        kernel_size=kernel_size,
                            ),
        STGCN_Fuse(
                    dim_in=128 // beta_inv,
                    A=A,
                    ),
        
        ))
        if edge_important_weight:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        # self.fcn = nn.Conv2d(512, cfg.MODEL.NUM_CLASSES, kernel_size=1)
        # self.fcn = nn.Linear(
        #             (128 // cfg.SLOWFAST.BETA_INV)*A.size(1) + 128 + 128 // cfg.SLOWFAST.BETA_INV, 
        #             cfg.MODEL.NUM_CLASSES
        #             )
        self.fcn = nn.Linear(
            (128+128 // beta_inv * A.size(1)) * (max_frame // beta_inv),
             num_class)

    def forward(self, x):
        for pathway in range(self.num_pathways):
            # data normalization
            N, C, T, V, M = x[pathway].size()
            x_ = x[pathway].permute(0, 4, 3, 1, 2).contiguous()
            x_ = x_.view(N * M, V * C, T)
            x_ = self.data_bn(x_)
            x_ = x_.view(N, M, V, C, T)
            x_ = x_.permute(0, 1, 3, 4, 2).contiguous()
            x_ = x_.view(N * M, C, T, V)
            x[pathway] = x_

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        output = x[0]
        T = output.size()[2]
        output = output.view(N*M, -1, V)
        output = F.avg_pool1d(output, output.size()[-1])   
        output.view(N ,M, -1, T, 1).mean(dim=1)    
        output = output.view(output.size(0), -1)
        
        # prediction
        output = self.fcn(output)


        return output

class SF_GCN(nn.Module):
    def __init__(
                self,
                num_pathways,
                in_channels,
                num_class, 
                max_frame,
                graph_args,
                beta_inv=8,
                alpha=8,
                edge_important_weight=True
                ):
        super().__init__()
        self.num_pathways = num_pathways

        # load graph
        self.graph = gcn_helper.Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.sf_gcn_networks = nn.ModuleList((
            gcn_helper.GCN(
                in_channels=[in_channels, in_channels],
                out_channels=[64, 64 // beta_inv],
                kernel_size=kernel_size[1]
            ),
            SFGCN_Fuse(
                dim_in=64 // beta_inv ,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[64 + 64 // beta_inv * A.size(1), 64 // beta_inv],
                out_channels=[64, 64 // beta_inv],
                kernel_size=kernel_size[1],
            ),
            SFGCN_Fuse(
                dim_in=64 // beta_inv,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[64 + 64 // beta_inv * A.size(1), 64 // beta_inv],
                out_channels=[64, 64 // beta_inv],
                kernel_size=kernel_size[1]
            ),
            SFGCN_Fuse(
                dim_in=64 // beta_inv,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[64 + 64 // beta_inv * A.size(1), 64 // beta_inv],
                out_channels=[64, 64 // beta_inv],
                kernel_size=kernel_size[1],
            ),
            SFGCN_Fuse(
                dim_in=64 // beta_inv,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[64 + 64 // beta_inv * A.size(1), 64 // beta_inv],
                out_channels=[128, 128 // beta_inv],
                kernel_size=kernel_size[1],
            ),
            SFGCN_Fuse(
                dim_in=128 // beta_inv,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[128 + 128 // beta_inv * A.size(1), 128 // beta_inv],
                out_channels=[128, 128 // beta_inv],
                kernel_size=kernel_size[1],
            ),
            SFGCN_Fuse(
                dim_in=128 // beta_inv,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[128 + 128 // beta_inv * A.size(1), 128 // beta_inv],
                out_channels=[128, 128 // beta_inv],
                kernel_size=kernel_size[1],
            ),
            SFGCN_Fuse(
                dim_in=128 // beta_inv,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[128 + 128 // beta_inv * A.size(1), 128 // beta_inv],
                out_channels=[256, 256 // beta_inv],
                kernel_size=kernel_size[1],
            ),
            SFGCN_Fuse(
                dim_in=256 // beta_inv,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[256 + 256 // beta_inv * A.size(1), 256 // beta_inv],
                out_channels=[256, 256 // beta_inv],
                kernel_size=kernel_size[1],
            ),
            SFGCN_Fuse(
                dim_in=256 // beta_inv,
                A=A,
            ),
            gcn_helper.GCN(
                in_channels=[256 + 256 // beta_inv * A.size(1), 256 // beta_inv],
                out_channels=[256, 256 // beta_inv],
                kernel_size=kernel_size[1],
            ),
            SFGCN_Fuse(
                dim_in=256 // beta_inv,
                A=A,
            ),
        ))

        if edge_important_weight:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.sf_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.sf_gcn_networks)

        self.fcn = nn.Linear(
            (128 + 128 // beta_inv * A.size(1))*(max_frame // alpha)*2,
             num_class
             )
        
        
    def forward(self, x):
        for pathway in range(self.num_pathways):
            # data normalization
            N, C, T, V, M = x[pathway].size()
            x_ = x[pathway].permute(0, 4, 3, 1, 2).contiguous()
            x_ = x_.view(N * M, V * C, T)
            x_ = self.data_bn(x_)
            x_ = x_.view(N, M, V, C, T)
            x_ = x_.permute(0, 1, 3, 4, 2).contiguous()
            x_ = x_.view(N * M, C, T, V)
            x[pathway] = x_  

        # forwad
        for gcn, importance in zip(self.sf_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance) 

        # pooling layer
        output = x[0]
        n, c, t, v = output.size()
        output = output.view(N*M, -1, v)
        output = F.avg_pool1d(output, output.size()[-1])
        output = output.view(N ,M, -1, t, 1).mean(dim=1)
        output = output.view(output.size(0), -1)

        # prediction
        output = self.fcn(output)
        # output = F.softmax(output, dim=1)


        return output
