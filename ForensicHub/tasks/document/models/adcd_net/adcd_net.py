"""
ADCD-Net: Adaptive DCT-Guided Content Disentangling Network for Document Tampering Detection

Paper: ICCV 2025
Reference: https://github.com/KAHIMWONG/ADCD-Net

This model combines:
1. Adaptive DCT modulation for handling block misalignment
2. Hierarchical content disentanglement for text-background separation
3. Pristine prototype estimation using background regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel

from .fph import FPH, AddCoord
from .restormer import get_restormer
from .loss import ADCDNetLoss, supcon_parallel


def get_mlp(in_channels, out_channels=None, bias=False):
    """Simple MLP with two 1x1 convolutions."""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels // 2, 1, bias=bias),
        nn.GELU(),
        nn.Conv2d(in_channels // 2, out_channels, 1, bias=bias)
    )


@register_model("ADCDNet")
class ADCDNet(BaseModel):
    """
    ADCD-Net for document tampering detection.

    Features:
    - Adaptive DCT modulation with alignment score prediction
    - Hierarchical content disentanglement
    - Pristine prototype estimation from background regions
    - Multi-scale feature extraction
    """

    def __init__(
            self,
            cls_n: int = 2,
            loc_out_dim: int = 96,
            rec_out_dim: int = 4,
            dct_feat_dim: int = 256,
            focal_in_dim: list = [192, 96, 96, 96],
            focal_out_dim: int = 32,
            pp_scale_n: int = 4,
            docres_ckpt_path: str = None,
            ce_weight: float = 3.0,
            rec_weight: float = 1.0,
            focal_weight: float = 0.2,
            norm_weight: float = 0.1,
            **kwargs
    ):
        """
        Args:
            cls_n: Number of classes (2 for binary segmentation)
            loc_out_dim: Output dimension for localization branch
            rec_out_dim: Output dimension for reconstruction (RGB + DCT)
            dct_feat_dim: DCT feature dimension
            focal_in_dim: Input dimensions for focal projection at each scale
            focal_out_dim: Output dimension for focal projection
            pp_scale_n: Number of scales for pristine prototype
            docres_ckpt_path: Path to pre-trained DocRes checkpoint
            ce_weight: Weight for cross-entropy loss
            rec_weight: Weight for reconstruction loss
            focal_weight: Weight for focal contrastive loss
            norm_weight: Weight for normalization loss
        """
        super().__init__()

        self.cls_n = cls_n
        self.docres_ckpt_path = docres_ckpt_path

        # RGB encoder + localization branch (Restormer-based)
        self.restormer_loc = get_restormer(model_name='full_model', out_channels=loc_out_dim)

        # Reconstruction branch (decoder only)
        self.restormer_rec = get_restormer(model_name='decoder_only', out_channels=rec_out_dim)

        # DCT encoder (Feature Pyramid Handler)
        self.dct_encoder = FPH(dct_feat_dim=dct_feat_dim)

        # Alignment score predictor
        self.dct_align_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dct_feat_dim, dct_feat_dim // 2, bias=False),
            nn.BatchNorm1d(dct_feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dct_feat_dim // 2, 2, bias=False)
        )

        # Coordinate addition for input
        self.add_coord = AddCoord(with_r=False)

        # Focal projection heads for contrastive learning
        self.focal_proj = nn.ModuleList([
            get_mlp(in_channels=focal_in_dim[i], out_channels=focal_out_dim)
            for i in range(len(focal_in_dim))
        ])

        # Pristine prototype estimation
        self.pp_scale_proj = get_mlp(in_channels=pp_scale_n, out_channels=loc_out_dim)
        self.pp_bias_proj = get_mlp(in_channels=pp_scale_n, out_channels=loc_out_dim)

        # Output head
        self.out_head = get_mlp(in_channels=loc_out_dim, out_channels=cls_n)

        # Loss function
        self.loss_fn = ADCDNetLoss(
            ce_weight=ce_weight,
            rec_weight=rec_weight,
            focal_weight=focal_weight,
            norm_weight=norm_weight
        )

        # Load pre-trained weights if available
        if docres_ckpt_path:
            self.load_docres()

    def load_docres(self):
        """Load pre-trained DocRes weights for the encoder."""
        try:
            ckpt = torch.load(self.docres_ckpt_path, map_location='cpu', weights_only=True)
            if 'model_state' in ckpt:
                ckpt = ckpt['model_state']
            elif 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']

            # Remove 'output' layer weights
            for name in list(ckpt.keys()):
                if 'output' in name:
                    ckpt.pop(name)

            # Remove 'module.' prefix if present
            new_ckpt = {}
            for key in ckpt:
                if key.startswith('module.'):
                    new_ckpt[key[7:]] = ckpt[key]
                else:
                    new_ckpt[key] = ckpt[key]

            miss, unexpected = self.restormer_loc.load_state_dict(new_ckpt, strict=False)
            print(f'Loaded DocRes checkpoint from {self.docres_ckpt_path}')
            if miss:
                print(f'  Missing keys: {len(miss)}')
            if unexpected:
                print(f'  Unexpected keys: {len(unexpected)}')
        except Exception as e:
            print(f'Warning: Could not load DocRes checkpoint: {e}')

    def get_pp_map(self, pp_feats, ocr_mask, img_size):
        """
        Compute pristine prototype maps from multi-scale features.

        Args:
            pp_feats: List of features at different scales
            ocr_mask: OCR mask indicating text regions (B, 1, H, W)
            img_size: Target output size

        Returns:
            pp_maps: Pristine prototype maps (B, num_scales, img_size, img_size)
        """
        maps_per_level = []

        # Use OCR mask for background identification
        y = ocr_mask.float()

        for f in pp_feats:
            f = F.normalize(f, p=2, dim=1)  # L2 normalize features
            _, c, h, w = f.shape

            # Resize mask to feature resolution
            bg_mask = (F.interpolate(y, size=(h, w), mode='nearest') == 0).float()

            # Compute background mean feature
            bg_sum = (f * bg_mask).sum(dim=(2, 3))
            bg_count = bg_mask.sum(dim=(2, 3)).clamp_min(1.0)
            bg_mean = F.normalize(bg_sum / bg_count, p=2, dim=1)

            # Compute cosine similarity map
            sim = (f * bg_mean[:, :, None, None]).sum(dim=1, keepdim=True)

            # Upscale to output size
            sim = F.interpolate(sim, size=(img_size, img_size), mode='bilinear', align_corners=False)
            maps_per_level.append(sim)

        return torch.cat(maps_per_level, dim=1)

    def forward(self, image, mask, dct=None, qt=None, ocr_mask=None, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            image: Input RGB image (B, 3, H, W)
            mask: Ground truth tampering mask (B, 1, H, W) or (B, H, W)
            dct: DCT coefficients (B, H, W) with values 0-20
            qt: Quantization table (B, 1, 8, 8) or (B, 8, 8)
            ocr_mask: OCR mask for text regions (B, 1, H, W)

        Returns:
            Dictionary containing:
            - backward_loss: Total loss for backpropagation
            - pred_mask: Predicted tampering mask
            - visual_loss: Individual losses for visualization
            - visual_image: Predicted mask for visualization
        """
        is_train = self.training
        img_size = image.size(2)

        # Ensure mask has correct shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask_for_loss = mask.squeeze(1).long()

        # Handle missing OCR mask
        if ocr_mask is None:
            ocr_mask = torch.zeros_like(mask)

        # Get alignment score & multi-scale DCT features
        if dct is not None and qt is not None:
            as_feat, ms_dct_feats = self.dct_encoder(dct, qt)
            align_logits = self.dct_align_head(as_feat)
            align_score = F.softmax(align_logits, dim=1)[:, -1]
        else:
            ms_dct_feats = None
            align_logits = None
            align_score = None

        # Prepare input with coordinates
        img_with_coord = self.add_coord(image)
        placeholder = torch.zeros_like(ocr_mask).float()
        img_input = torch.cat([img_with_coord, placeholder], dim=1)

        # Get RGB features & localization
        feat, cnt_feats, frg_feats, pp_feats = self.restormer_loc(
            img=img_input,
            ms_dct_feats=ms_dct_feats,
            dct_align_score=align_score
        )

        # Compute pristine prototype maps
        pp_maps = self.get_pp_map(pp_feats, ocr_mask, img_size)
        pp_scale = self.pp_scale_proj(pp_maps)
        pp_bias = self.pp_bias_proj(pp_maps)

        # Apply pristine prototype modulation
        feat = feat * pp_scale + pp_bias

        # Final output
        logits = self.out_head(feat)

        # Training-specific computations
        rec_items = None
        focal_losses = None

        if is_train:
            # Reconstruction with shuffled features
            shuffle_rec_img = self.restormer_rec(cnt_feats, frg_feats, is_shuffle=True)
            if dct is not None:
                norm_dct = (dct.float() / 20.0).unsqueeze(1)
                rec_items = (shuffle_rec_img, norm_dct)

            # Focal contrastive losses
            focal_losses = tuple([
                supcon_parallel(self.focal_proj[i](pp_feats[i]), mask)
                for i in range(len(pp_feats))
            ])

        # Compute loss
        total_loss, loss_dict = self.loss_fn(
            logits=logits,
            mask=mask_for_loss,
            align_logits=align_logits,
            rec_items=rec_items,
            focal_losses=focal_losses
        )

        # Get predicted mask
        pred_mask = F.softmax(logits, dim=1)[:, 1:]  # Take forgery class probability

        output_dict = {
            "backward_loss": total_loss,
            "pred_mask": pred_mask,
            "visual_loss": {
                "seg_loss": loss_dict.get('seg_loss', torch.tensor(0.0)),
                "align_loss": loss_dict.get('align_loss', torch.tensor(0.0)),
                "rec_loss": loss_dict.get('rec_loss', torch.tensor(0.0)),
                "focal_loss": loss_dict.get('focal_loss', torch.tensor(0.0)),
                "combined_loss": total_loss,
            },
            "visual_image": {
                "pred_mask": pred_mask,
            }
        }

        return output_dict

    def get_prediction(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference without loss computation.

        Args:
            data_dict: Dictionary containing input data

        Returns:
            Dictionary with pred_mask
        """
        self.eval()
        with torch.no_grad():
            image = data_dict['image']
            dct = data_dict.get('dct', None)
            qt = data_dict.get('qt', None)
            ocr_mask = data_dict.get('ocr_mask', None)

            img_size = image.size(2)

            if ocr_mask is None:
                ocr_mask = torch.zeros(image.size(0), 1, image.size(2), image.size(3), device=image.device)

            # DCT features
            if dct is not None and qt is not None:
                as_feat, ms_dct_feats = self.dct_encoder(dct, qt)
                align_logits = self.dct_align_head(as_feat)
                align_score = F.softmax(align_logits, dim=1)[:, -1]
            else:
                ms_dct_feats = None
                align_score = None

            # Prepare input
            img_with_coord = self.add_coord(image)
            placeholder = torch.zeros_like(ocr_mask).float()
            img_input = torch.cat([img_with_coord, placeholder], dim=1)

            # Forward pass
            feat, cnt_feats, frg_feats, pp_feats = self.restormer_loc(
                img=img_input,
                ms_dct_feats=ms_dct_feats,
                dct_align_score=align_score
            )

            # Pristine prototype
            pp_maps = self.get_pp_map(pp_feats, ocr_mask, img_size)
            pp_scale = self.pp_scale_proj(pp_maps)
            pp_bias = self.pp_bias_proj(pp_maps)

            feat = feat * pp_scale + pp_bias
            logits = self.out_head(feat)

            pred_mask = F.softmax(logits, dim=1)[:, 1:]

            return {
                "pred_mask": pred_mask,
            }


if __name__ == "__main__":
    # Test the model
    model = ADCDNet()
    model.train()

    # Create dummy inputs
    batch_size = 2
    img_size = 256
    image = torch.randn(batch_size, 3, img_size, img_size)
    mask = torch.randint(0, 2, (batch_size, 1, img_size, img_size))
    dct = torch.randint(0, 21, (batch_size, img_size, img_size))
    qt = torch.randint(0, 64, (batch_size, 1, 8, 8))
    ocr_mask = torch.randint(0, 2, (batch_size, 1, img_size, img_size))

    # Forward pass
    output = model(image=image, mask=mask, dct=dct, qt=qt, ocr_mask=ocr_mask)

    print("Output keys:", output.keys())
    print("Backward loss:", output['backward_loss'].item())
    print("Pred mask shape:", output['pred_mask'].shape)
    print("Visual losses:", {k: v.item() if torch.is_tensor(v) else v for k, v in output['visual_loss'].items()})