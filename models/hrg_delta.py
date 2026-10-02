import torch
import torch.nn as nn
import torch.nn.functional as F

class HRGDeltaLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        rank: int = 4,
        groups: int = 4,
        dynamic_q: bool = True,
        dynamic_beta: bool = True,
        beta_max: float = 2.0,
        value_map_type: str = "grouped_conv",
        q_mode: str = "qr",
        eps: float = 1e-8,
    ):
        super().__init__()
        
        assert in_channels % groups == 0
        assert rank <= in_channels // groups
        assert beta_max > 0.0
        assert q_mode == "qr"
        
        self.in_channels = in_channels
        self.rank = rank
        self.groups = groups
        self.group_dim = in_channels // groups
        self.dynamic_q = dynamic_q
        self.dynamic_beta = dynamic_beta
        self.beta_max = beta_max
        self.q_mode = q_mode
        self.eps = eps
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        if dynamic_q:
            self.q_gen = nn.Linear(
                in_channels,
                groups * self.group_dim * rank,
            )
        else:
            self.q_raw_static = nn.Parameter(
                torch.randn(groups, self.group_dim, rank) * 0.02
            )
            
        if dynamic_beta:
            self.beta_gen = nn.Linear(in_channels, groups * rank)
            nn.init.constant_(self.beta_gen.bias, -1.0)
        else:
            self.beta_raw_static = nn.Parameter(
                torch.full((groups, rank), -1.0)
            )
            
        self.value_map = self._build_value_map(in_channels, groups)
        
        # Diagnostics
        self.last_beta = None
        self.last_beta_mean = None
        self.last_beta_max = None
        self.last_hyper_fraction = None
        self.last_orthogonality_error = None

    def _build_value_map(self, in_channels: int, groups: int):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def _get_q(self, descriptor):
        batch_size = descriptor.size(0)
        if self.dynamic_q:
            raw_q = self.q_gen(descriptor)
            raw_q = raw_q.view(batch_size, self.groups, self.group_dim, self.rank)
            q, _ = torch.linalg.qr(raw_q, mode="reduced")
        else:
            raw_q = self.q_raw_static.unsqueeze(0).expand(batch_size, -1, -1, -1)
            q, _ = torch.linalg.qr(raw_q, mode="reduced")
            
        # Diagnostics: orthogonality error
        with torch.no_grad():
            qt_q = torch.einsum("bgdi,bgdj->bgij", q, q)
            eye = torch.eye(self.rank, device=q.device).view(1, 1, self.rank, self.rank)
            self.last_orthogonality_error = torch.linalg.matrix_norm(qt_q - eye, ord="fro").mean().item()
            
        return q

    def _get_beta(self, descriptor):
        batch_size = descriptor.size(0)
        if self.dynamic_beta:
            raw_beta = self.beta_gen(descriptor)
            raw_beta = raw_beta.view(batch_size, self.groups, self.rank)
        else:
            raw_beta = self.beta_raw_static.unsqueeze(0).expand(batch_size, -1, -1)
            
        beta = self.beta_max * torch.sigmoid(raw_beta)
        
        # Diagnostics
        with torch.no_grad():
            self.last_beta = beta.detach()
            self.last_beta_mean = beta.mean().item()
            self.last_beta_max = beta.max().item()
            self.last_hyper_fraction = (beta > 1.0).float().mean().item()
            
        return beta

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        spatial_size = height * width

        descriptor = self.pool(x).flatten(1)
        q = self._get_q(descriptor)
        beta = self._get_beta(descriptor)
        v_ambient = self.value_map(x)

        x_grouped = x.reshape(
            batch_size,
            self.groups,
            self.group_dim,
            spatial_size,
        )
        v_grouped = v_ambient.reshape(
            batch_size,
            self.groups,
            self.group_dim,
            spatial_size,
        )

        residual_ambient = v_grouped - x_grouped
        residual_subspace = torch.einsum(
            "bgdr,bgds->bgrs",
            q,
            residual_ambient,
        )
        gated_subspace = beta.unsqueeze(-1) * residual_subspace
        update = torch.einsum(
            "bgdr,bgrs->bgds",
            q,
            gated_subspace,
        )

        update = update.reshape(batch_size, channels, height, width)
        return x + update
