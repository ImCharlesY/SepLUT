
import torch
import torch.nn as nn
import torch.nn.functional as F


def lut_transform(imgs, luts):
    # img (b, 3, h, w), lut (b, c, m, m, m)

    # normalize pixel values
    imgs = (imgs - .5) * 2.
    # reshape img to grid of shape (b, 1, h, w, 3)
    grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)

    # after gridsampling, output is of shape (b, c, 1, h, w)
    outs = F.grid_sample(luts, grids,
        mode='bilinear', padding_mode='border', align_corners=True)
    # remove the extra dimension
    outs = outs.squeeze(2)
    return outs


class LUT1DGenerator(nn.Module):
    r"""The 1DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points.
        n_feats (int): Dimension of the input image representation vector.
        color_share (bool, optional): Whether to share a single 1D LUT across
            three color channels. Default: False.
    """

    def __init__(self, n_colors, n_vertices, n_feats, color_share=False) -> None:
        super().__init__()
        repeat_factor = n_colors if not color_share else 1
        self.lut1d_generator = nn.Linear(
            n_feats, n_vertices * repeat_factor)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.color_share = color_share

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        lut1d = self.lut1d_generator(x).view(
            x.shape[0], -1, self.n_vertices)
        if self.color_share:
            lut1d = lut1d.repeat_interleave(self.n_colors, dim=1)
        lut1d = lut1d.sigmoid()
        return lut1d


class LUT3DGenerator(nn.Module):
    r"""The 3DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity