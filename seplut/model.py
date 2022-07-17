#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2022 Charles
'''

import numbers
import os.path as osp

import torch
import torch.nn as nn

import mmcv
from mmcv.runner import auto_fp16
from mmcv.utils import get_logger

from mmedit.models.base import BaseModel
from mmedit.models.registry import MODELS
from mmedit.models.builder import build_loss
from mmedit.core import psnr, ssim, tensor2img

from .modules.backbone import LightBackbone, Res18Backbone
from .modules.lut import lut_transform, LUT1DGenerator, LUT3DGenerator
from seplut_ext import seplut_transform


__all__ = ['SepLUT']


@MODELS.register_module()
class SepLUT(BaseModel):
    r"""Separable Image-adaptive Lookup Tables for Real-time Image Enhancement.

    Args:
        n_ranks (int, optional): Number of ranks for 3D LUT (or the number of basis
            LUTs). Default: 3.
        n_vertices_3d (int, optional): Size of the 3D LUT. If `n_vertices_3d` <= 0,
            the 3D LUT will be disabled. Default: 17.
        n_vertices_1d (int, optional): Size of the 1D LUTs. If `n_vertices_1d` <= 0,
            the 1D LUTs will be disabled. Default: 17.
        lut1d_color_share (bool, optional): Whether to share a single 1D LUT across
            three color channels. Default: False.
        backbone (str, optional): Backbone architecture to use. Can be either 'light'
            or 'res18'. Default: 'light'.
        n_base_feats (int, optional): The channel multiplier of the backbone network.
            Only used when `backbone` is 'light'. Default: 8.
        pretrained (bool, optional): Whether to use ImageNet-pretrained weights.
            Only used when `backbone` is 'res18'. Default: None.
        n_colors (int, optional): Number of input color channels. Default: 3.
        sparse_factor (float, optional): Loss weight for the sparse regularization term.
            Default: 0.0001.
        smooth_factor (float, optional): Loss weight for the smoothness regularization term.
            Default: 0.
        monotonicity_factor (float, optional): Loss weight for the monotonicaity
            regularization term. Default: 10.0.
        recons_loss (dict, optional): Config for pixel-wise reconstruction loss.
        train_cfg (dict, optional): Config for training. Default: None.
        test_cfg (dict, optional): Config for testing. Default: None.
    """

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}
    # quantization_mode: (n_vertices_1d, n_vertices_3d)
    allowed_quantization_modes = {(9, 9), (17, 17)}

    def __init__(self,
        n_ranks=3,
        n_vertices_3d=17,
        n_vertices_1d=17,
        lut1d_color_share=False,
        backbone='light',
        n_base_feats=8,
        pretrained=False,
        n_colors=3,
        sparse_factor=0.0001,
        smooth_factor=0,
        monotonicity_factor=0,
        recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
        train_cfg=None,
        test_cfg=None):

        super().__init__()

        assert backbone in ['light', 'res18']
        assert n_vertices_3d > 0 or n_vertices_1d > 0

        self.backbone = dict(
            light=LightBackbone,
            res18=Res18Backbone)[backbone.lower()](
                pretrained=pretrained,
                extra_pooling=True,
                n_base_feats=n_base_feats)

        if n_vertices_3d > 0:
            self.lut3d_generator = LUT3DGenerator(
                n_colors, n_vertices_3d, self.backbone.out_channels, n_ranks)

        if n_vertices_1d > 0:
            self.lut1d_generator = LUT1DGenerator(
                n_colors, n_vertices_1d, self.backbone.out_channels,
                color_share=lut1d_color_share)

        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices_3d = n_vertices_3d
        self.n_vertices_1d = n_vertices_1d
        self.sparse_factor = sparse_factor
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor
        self.backbone_name = backbone.lower()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False

        self.init_weights()

        self.recons_loss = build_loss(recons_loss)

        # variables for quantization
        self.en_quant = test_cfg.get('en_quant', False) if test_cfg else False
        self.quantization_mode = (self.n_vertices_1d, self.n_vertices_3d)
        self._quantized = False
        if self.en_quant and self.quantization_mode not in self.allowed_quantization_modes:
            get_logger('seplut').warning('Current implementation does not support '
                'quantization on mode 1D#{}-3D#{}. Quantization is disabled.'.format(
                    *self.quantization_mode))
            self.en_quant = False

    def init_weights(self):
        r"""Init weights for models.

        For the backbone network and the 3D LUT generator, we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
        """
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        if self.backbone_name not in ['res18']:
            self.apply(special_initilization)
        if self.n_vertices_3d > 0:
            self.lut3d_generator.init_weights()

    def quantize(self):
        r'''Apply PyTorch's dynamic quantization technique to model parameters.
        '''
        if not self.en_quant or self._quantized: return
        if 'cuda' in str(next(self.parameters()).device):
            get_logger('seplut').warning('Current implementation does not support '
                'quantization on GPU model. Quantization is disabled. Please run '
                'the inference on CPU.')
            self.en_quant = False
            return
        self.modules_backup = {
            self.lut1d_generator, self.lut3d_generator}
        self.lut1d_generator = torch.quantization.quantize_dynamic(
            self.lut1d_generator, {nn.Linear}, dtype=torch.qint8)
        self.lut3d_generator = torch.quantization.quantize_dynamic(
            self.lut3d_generator, {nn.Linear}, dtype=torch.qint8)
        self._quantized = True

    def dequantize(self):
        r'''Restore model parameters after model quantization.
        '''
        if not self._quantized: return
        self.lut1d_generator, self.lut3d_generator = self.modules_backup
        del self.modules_backup
        self._quantized = False

    def preprocess_quantized_transform(self, img, lut1d, lut3d):
        r'''Quantize input image, 1D LUT and 3D LUT into 8-bit representation.

        Args:
            img (Tensor): Input image, shape (b, c, h, w).
            lut1d (Tensor): 1D LUT, shape (b, c, n_vertices_1d).
            lut3d (Tensor): 3D LUT, shape
                (b, c, n_vertices_3d, n_vertices_3d, n_vertices_3d).
        Returns:
            tuple(Tensor, Tensor, Tensor, float, float):
                Quantized input image, 1D LUT, 3D LUT,
                minimum and maximum values of the 3D LUT.
        '''
        lmin, lmax = lut3d.min(), lut3d.max()
        if self._quantized:
            img = img.mul(255).round().to(torch.uint8)
            lut1d = lut1d.mul(255).round().to(torch.uint8)
            lut3d = lut3d.sub(lmin).div(lmax - lmin)
            lut3d = lut3d.mul(255).round().to(torch.uint8)
        return img, lut1d, lut3d, lmin, lmax

    def postprocess_quantized_transform(self, out, lmin, lmax):
        r'''Dequantize output image.

        Args:
            out (Tensor): Output image, shape (b, c, h, w).
            lmin (float): minimum float value in the original 3D LUT.
            lmax (float): maximum float value in the original 3D LUT.
        Returns:
            Tensor: Dequantized output image.
        '''
        if self._quantized:
            out = out.float().div(255)
            out = out.float().mul(lmax - lmin).add(lmin).clamp(0, 1)
            out = out.mul(255).round().div(255)
        return out

    def forward_dummy(self, imgs):
        r"""The real implementation of model forward.

        Args:
            img (Tensor): Input image, shape (b, c, h, w).
        Returns:
            tuple(Tensor, Tensor, Tensor):
                Output image, 3DLUT weights, 1DLUTs.
        """
        # context vector: (b, f)
        codes = self.backbone(imgs)

        # generate 3x 1DLUTs and perform the 1D LUT transform
        if self.n_vertices_1d > 0:
            # (b, c, m)
            lut1d = self.lut1d_generator(codes)
            # achieved by converting the 1DLUTs into equivalent 3DLUT
            iluts = []
            for i in range(imgs.shape[0]):
                iluts.append(torch.stack(
                    torch.meshgrid(*(lut1d[i].unbind(0)[::-1])),
                    dim=0).flip(0))
            # (b, c, m, m, m)
            iluts = torch.stack(iluts, dim=0)
            imgs = lut_transform(imgs, iluts)
        else:
            lut1d = imgs.new_zeros(1)

        # generate 3DLUT and perform the 3D LUT transform
        if self.n_vertices_3d > 0:
            # (b, c, d, d, d)
            lut3d_weights, lut3d = self.lut3d_generator(codes)
            outs = lut_transform(imgs, lut3d)
        else:
            lut3d_weights = imgs.new_zeros(1)
            outs = imgs

        return outs, lut3d_weights, lut1d

    def forward_fast(self, imgs):
        r"""The fast implementation of model forward. It uses a custom PyTorch
        extension `seplut_transform` that merges the 1D and 3D LUT transforms
        into a single kernel for efficiency.

        [NOTE] The backward function of `seplut_transform` is not implemented,
               so it cannot be used in the training.

        Args:
            img (Tensor): Input image, shape (b, c, h, w).
        Returns:
            Tensor: Output image.
        """
        self.quantize()

        # context vector: (b, f)
        codes = self.backbone(imgs)

        # 3x 1DLUTs: (b, c, m)
        if self.n_vertices_1d > 0:
            lut1d = self.lut1d_generator(codes)
        else:
            lut1d = (torch.arange(4, device=imgs.device)
                        .div(3).repeat(self.n_colors, 1))
            lut1d = lut1d.unsqueeze(0).repeat(imgs.shape[0], 1, 1)

        # 3DLUT: (b, c, d, d, d)
        if self.n_vertices_3d > 0:
            _, lut3d = self.lut3d_generator(codes)
        else:
            lut3d = torch.stack(
                torch.meshgrid(*[torch.arange(4, device=imgs.device) \
                    for _ in range(self.n_colors)]),
                dim=0).div(3).flip(0)
            lut3d = lut3d.unsqueeze(0).repeat(
                imgs.shape[0], 1, *([1] * self.n_colors))

        imgs, lut1d, lut3d, lmin, lmax = \
            self.preprocess_quantized_transform(imgs, lut1d, lut3d)
        out = seplut_transform(imgs, lut3d, lut1d)
        out = self.postprocess_quantized_transform(out, lmin, lmax)

        self.dequantize()

        return out

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        r"""Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor, optional): Ground-truth image. Default: None.
            test_mode (bool, optional): Whether in test mode or not. Default: False.
            kwargs (dict, optional): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        r"""Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            outputs (dict): Output results.
        """
        losses = dict()
        output, lut3d_weights, lut1d = self.forward_dummy(lq)
        losses['loss_recons'] = self.recons_loss(output, gt)
        if self.sparse_factor > 0 and lut3d_weights is not None:
            losses['loss_sparse'] = self.sparse_factor * torch.mean(lut3d_weights.pow(2))
        if self.n_vertices_3d > 0:
            reg_smoothness, reg_monotonicity = self.lut3d_generator.regularizations(
                self.smooth_factor, self.monotonicity_factor)
            if self.smooth_factor > 0:
                losses['loss_smooth'] = reg_smoothness
            if self.monotonicity_factor > 0:
                losses['loss_mono'] = reg_monotonicity
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        r"""Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor, optional): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool, optional): Whether to save image. Default: False.
            save_path (str, optional): Path to save image. Default: None.
            iteration (int, optional): Iteration for the saving image name.
                Default: None.
        Returns:
            outputs (dict): Output results.
        """
        output = self.forward_fast(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def train_step(self, data_batch, optimizer):
        r"""Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.
        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.update({'log_vars': log_vars})

        return outputs

    def val_step(self, data_batch, **kwargs):
        r"""Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict, optional): Other arguments for ``val_step``.
        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def evaluate(self, output, gt):
        r"""Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](
                output, gt, crop_border)
        return eval_result
