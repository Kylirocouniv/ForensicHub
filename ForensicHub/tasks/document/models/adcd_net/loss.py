"""
Loss functions for ADCD-Net document tampering detection.

Includes:
- Supervised Contrastive Loss (Focal Loss variant)
- Soft Cross-Entropy Loss with label smoothing
- Lovasz Loss for segmentation
- Combined loss for ADCD-Net training
"""

from typing import Optional
from itertools import filterfalse as ifilterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# ========================
# Supervised Contrastive Loss
# ========================

def supcon_parallel(f, y, t=0.1, sample_n=512, min_n=3):
    """
    Supervised contrastive loss computed in parallel across batch.

    Args:
        f: Feature maps (B, C, H, W)
        y: Segmentation masks (B, 1, H, W) or (B, H, W)
        t: Temperature parameter
        sample_n: Number of samples per class
        min_n: Minimum samples required per class

    Returns:
        Contrastive loss scalar
    """
    b, c, h, w = f.shape

    # Reshape y to match f's spatial size
    if y.dim() == 4:
        y = y.squeeze(1)
    y = F.interpolate(y.float().unsqueeze(1), size=(h, w), mode='nearest').long().squeeze(1)

    l = h * w
    f = f.permute(0, 2, 3, 1).reshape(b, l, c)
    y = y.reshape(b, l)

    # Sample features per class
    f_list, y_list, len_list = [], [], []
    for b_idx in range(b):
        bf = f[b_idx]
        by = y[b_idx]
        r_f = bf[by == 0]  # Real/pristine features
        f_f = bf[by == 1]  # Forged features
        r_n, f_n = r_f.size(0), f_f.size(0)

        if r_n < min_n or f_n < min_n:
            continue

        sample_r_f = r_f[torch.randperm(r_f.size(0))[:sample_n]]
        sample_f_f = f_f[torch.randperm(f_f.size(0))[:sample_n]]

        sample_f = torch.cat([sample_r_f, sample_f_f], 0)
        sample_y = torch.cat([torch.zeros(sample_r_f.size(0)), torch.ones(sample_f_f.size(0))], 0)

        f_list.append(sample_f)
        y_list.append(sample_y)
        len_list.append(sample_f.size(0))

    if len(f_list) == 0:
        return torch.tensor([0.0], device=f.device, requires_grad=True).sum()

    pad_f = pad_sequence(f_list, batch_first=True, padding_value=1)
    y = pad_sequence(y_list, batch_first=True, padding_value=-1).to(f.device)
    is_pad = pad_f.sum(-1) == c
    f = F.normalize(pad_f, dim=-1)
    b, l, c = pad_f.shape

    sim = torch.bmm(f, f.permute(0, 2, 1))
    sim = torch.exp(torch.div(sim, t))

    is_valid = ~is_pad
    valid_mask = torch.bmm(is_valid[:, :, None].float(), is_valid[:, None, :].float())

    p_mask = (y[:, None, :] == y[:, :, None]).float()
    eyes = torch.eye(l, dtype=torch.bool, device=f.device).repeat(b, 1, 1)
    reverse_eyes = ~eyes
    p_mask = p_mask * reverse_eyes * valid_mask
    p_num = torch.sum(p_mask, dim=-1)

    n_mask = (y[:, None, :] != y[:, :, None]).float()
    n_mask = n_mask * reverse_eyes * valid_mask

    denominator_p = torch.sum(sim * p_mask, dim=-1, keepdim=True)
    denominator_n = torch.sum(sim * n_mask, dim=1, keepdim=True)
    denominator = denominator_p + denominator_n

    logits = torch.sum(torch.log(sim / (denominator + 1e-8)) * p_mask, dim=-1)
    logits = (-logits / (p_num + 1e-8))
    logits = logits.mean()

    return logits


# ========================
# Soft Cross-Entropy Loss
# ========================

def label_smoothed_nll_loss(
        lprobs: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        ignore_index=None,
        reduction="mean",
        dim=-1
) -> torch.Tensor:
    """
    Label smoothed negative log likelihood loss.

    Args:
        lprobs: Log-probabilities of predictions
        target: Target labels
        epsilon: Label smoothing factor
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'
        dim: Dimension for gathering
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class SoftCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(
            self,
            reduction: str = "mean",
            smooth_factor: Optional[float] = 0.1,
            ignore_index: Optional[int] = -100,
            dim: int = 1,
    ):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )


# ========================
# Lovasz Loss
# ========================

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def _lovasz_grad(gt_sorted):
    """Compute gradient of Lovasz extension."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """Binary Lovasz hinge loss."""
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss, flat version."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flatten predictions and labels."""
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def _lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss."""
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def _lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss, flat version."""
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).type_as(probas)
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flatten probabilities and labels for Lovasz loss."""
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    C = probas.size(1)
    probas = torch.movedim(probas, 0, -1)
    probas = probas.contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Compute mean, handling NaN values."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(nn.Module):
    """Lovasz loss for image segmentation."""

    def __init__(
            self,
            mode: str = MULTICLASS_MODE,
            per_image: bool = False,
            ignore_index: Optional[int] = None,
            from_logits: bool = True,
    ):
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.from_logits = from_logits

    def forward(self, y_pred, y_true):
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            loss = _lovasz_hinge(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        elif self.mode == MULTICLASS_MODE:
            if self.from_logits:
                y_pred = y_pred.softmax(dim=1)
            loss = _lovasz_softmax(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        else:
            raise ValueError(f"Wrong mode {self.mode}")
        return loss


# ========================
# Combined ADCD-Net Loss
# ========================

class ADCDNetLoss(nn.Module):
    """
    Combined loss for ADCD-Net training.

    Combines:
    - Segmentation loss (CE + Lovasz)
    - Alignment loss (CE for alignment score prediction)
    - Reconstruction loss (L1)
    - Focal contrastive loss
    """

    def __init__(
            self,
            ce_weight: float = 3.0,
            rec_weight: float = 1.0,
            focal_weight: float = 0.2,
            norm_weight: float = 0.1,
            smooth_factor: float = 0.1,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.rec_weight = rec_weight
        self.focal_weight = focal_weight
        self.norm_weight = norm_weight

        self.seg_ce = SoftCrossEntropyLoss(smooth_factor=smooth_factor)
        self.lovasz = LovaszLoss(mode=MULTICLASS_MODE, per_image=True)
        self.align_ce = nn.CrossEntropyLoss()

    def forward(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            align_logits: torch.Tensor = None,
            rec_items: tuple = None,
            focal_losses: tuple = None,
    ):
        """
        Compute combined loss.

        Args:
            logits: Segmentation logits (B, 2, H, W)
            mask: Ground truth mask (B, H, W) or (B, 1, H, W)
            align_logits: Alignment prediction logits (B, 2)
            rec_items: Tuple of (reconstructed, target) for reconstruction loss
            focal_losses: Tuple of focal contrastive losses at multiple scales

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses for logging
        """
        # Ensure mask has correct shape
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        mask = mask.long()

        # Resize logits to match mask if needed
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)

        # Segmentation loss
        seg_ce_loss = self.seg_ce(logits, mask)
        seg_lovasz_loss = self.lovasz(logits, mask)
        seg_loss = seg_ce_loss + seg_lovasz_loss

        total_loss = self.ce_weight * seg_loss
        loss_dict = {
            'seg_ce_loss': seg_ce_loss,
            'seg_lovasz_loss': seg_lovasz_loss,
            'seg_loss': seg_loss,
        }

        # Alignment loss
        if align_logits is not None:
            # Create alignment targets: 1 if image has tampering, 0 otherwise
            align_target = (mask.sum(dim=(1, 2)) > 0).long()
            align_loss = self.align_ce(align_logits, align_target)
            total_loss = total_loss + align_loss
            loss_dict['align_loss'] = align_loss

        # Reconstruction loss
        if rec_items is not None:
            rec_pred, rec_target = rec_items
            # Split into RGB and DCT components
            rec_rgb = rec_pred[:, :3]
            rec_dct = rec_pred[:, 3:]

            # L1 loss for reconstruction
            if rec_target.dim() == 4 and rec_target.shape[1] == 1:
                # DCT-only target
                rec_loss = F.l1_loss(rec_dct, rec_target)
            else:
                # Full reconstruction
                rec_loss = F.l1_loss(rec_pred, rec_target)

            total_loss = total_loss + self.rec_weight * rec_loss
            loss_dict['rec_loss'] = rec_loss

        # Focal contrastive loss
        if focal_losses is not None:
            focal_loss = sum(focal_losses) / len(focal_losses)
            total_loss = total_loss + self.focal_weight * focal_loss
            loss_dict['focal_loss'] = focal_loss

        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict