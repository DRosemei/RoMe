import torch
import torch.nn as nn


class SmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, depth):
        grad_disp_x = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        grad_disp_y = torch.abs(depth[:, :-1, :, :] - depth[:, 1:, :, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), -1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :-1, :, :] - img[:, 1:, :, :]), -1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


class L1MaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask):
        loss = self.loss_fn(pred, target)
        loss = loss * mask
        return loss


class MESMaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask):
        loss = self.loss_fn(pred, target)
        loss = loss * mask
        return loss


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        SSIM_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        masked_SSIM = SSIM_loss * mask
        return masked_SSIM


class CELossWithMask(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target, mask):
        """
        pred and target are Variables.
        Assumes the inputs are in range [0, 1].
        """
        if self.weight is not None:
            weight = self.weight
        else:
            weight = torch.ones_like(pred[0])

        if self.reduction == 'mean':
            weight = weight.mean()

        loss = nn.CrossEntropyLoss(weight=weight, reduction=self.reduction)(pred, target)
        loss *= mask
        loss = loss.mean()
        return loss
