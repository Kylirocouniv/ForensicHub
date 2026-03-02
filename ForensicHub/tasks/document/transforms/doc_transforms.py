import pickle
from typing import List
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from ForensicHub.core.base_transform import BaseTransform
from ForensicHub.registry import register_transform
from ForensicHub.common.transforms import PillowJpegCompression, JpegCompressionWithDCT
@register_transform("DocTransform")
class DocTransform(BaseTransform):
    """Transform class for Doc tasks."""

    def __init__(self,luminance_path="",chrominance_path="", output_size: tuple = (512, 512), norm_type='image_net',compression_type="cv"):
        super().__init__()
        self.output_size = output_size
        self.norm_type = norm_type
        self.compression_type = compression_type
        try:
            self.matrice_luminance = self._load_quantization_tables(luminance_path)
            self.matrice_chrominance = self._load_quantization_tables(chrominance_path)
        except:
            pass

    def _load_quantization_tables(self, file_path: str) -> List:
        """Charge les tables de quantification depuis un fichier pickle.

        Args:
            file_path: Chemin vers le fichier pickle contenant les tables

        Returns:
            Liste de tables de quantification
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def get_post_transform(self) -> albu.Compose:
        """Get post-processing transforms like normalization and conversion to tensor."""
        if self.norm_type == 'image_net':
            return albu.Compose([
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'clip':
            return albu.Compose([
                albu.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'standard':
            return albu.Compose([
                albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'none':
            return albu.Compose([
                albu.ToFloat(max_value=255.0),  # 确保 uint8 转 float32，并映射到 [0, 1]
                ToTensorV2(transpose_mask=True)
            ])
        else:
            raise NotImplementedError("Normalization type not supported, use image_net, clip, standard or none")

    def get_train_transform(self) -> albu.Compose:
        """Get training transforms."""
        if self.compression_type == "cv":
            self.jpeg_compression = JpegCompressionWithDCT(
                quality_range=(30, 100),
                p=1.0,
            )
            return albu.Compose([
                # Flips
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                # Brightness and contrast fluctuation
                albu.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=0.1,
                    p=1
                ),
                self.jpeg_compression,
                # Rotate
                albu.RandomRotate90(p=0.5),
                # Blur
                albu.GaussianBlur(
                    blur_limit=(3, 7),
                    p=0.2
                )
            ])
        elif self.compression_type == "pillow":
            self.jpeg_compression = PillowJpegCompression(
                luma_tables=self.matrice_luminance,
                chroma_tables=self.matrice_chrominance,
                p=1.0,
            )
            return albu.Compose([
                # Flips
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                # Brightness and contrast fluctuation
                albu.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=0.1,
                    p=1
                ),
                self.jpeg_compression,
                # Rotate
                albu.RandomRotate90(p=0.5),
                # Blur
                albu.GaussianBlur(
                    blur_limit=(3, 7),
                    p=0.2
                )
            ])
        else:
            raise ValueError

    def get_test_transform(self) -> albu.Compose:
        """Get testing transforms."""
        return albu.Compose([
        ])
