import os
import random
import tempfile
from io import BytesIO

import cv2
import jpegio
import numpy as np
from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform


class PillowJpegCompression(ImageOnlyTransform):
    """Apply JPEG compression using Pillow with custom quantization tables.

    Unlike quality-based compression, this transform accepts explicit quantization
    tables for the luminance and chrominance channels, giving fine-grained control
    over the compression artefacts introduced.

    At each application a table is drawn at random from ``luma_tables`` and
    independently from ``chroma_tables``.  The image is encoded to JPEG in memory
    using those tables and immediately decoded back to RGB, replicating what happens
    when an image is saved/re-opened with those settings.

    After each call the DCT coefficients and quantization table produced by the
    **same** compression pass are available as:

    - ``self._last_dct``  – int32 array of quantised DCT coefficients (Y channel)
    - ``self._last_qtb``  – int32 8×8 luminance quantisation table

    Args:
        luma_tables (list[list[int]]): Non-empty list of luminance quantization
            tables.  Each table is a flat list of 64 integers that represent the
            8×8 DCT quantization matrix stored in JPEG zigzag order.
        chroma_tables (list[list[int]]): Non-empty list of chrominance
            quantization tables in the same format as ``luma_tables``.
        always_apply (bool): Whether to always apply the transform regardless of
            ``p``.  Defaults to ``False``.
        p (float): Probability of applying the transform.  Defaults to ``0.5``.

    Raises:
        ValueError: If either table list is empty or any table does not contain
            exactly 64 values.

    Example::

        import albumentations as albu
        from ForensicHub.common.transforms import PillowJpegCompression

        # Standard JPEG luminance quantization table (quality ≈ 50)
        luma_q50 = [
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68,109,103, 77,
            24, 35, 55, 64, 81,104,113, 92,
            49, 64, 78, 87,103,121,120,101,
            72, 92, 95, 98,112,100,103, 99,
        ]

        # High-quality luminance table (quality ≈ 90)
        luma_q90 = [
             3,  2,  2,  3,  5,  8, 10, 12,
             2,  2,  3,  4,  5, 12, 12, 11,
             3,  3,  3,  5,  8, 11, 14, 11,
             3,  3,  4,  6, 10, 17, 16, 12,
             4,  4,  7, 11, 14, 22, 21, 15,
             5,  7, 11, 13, 16, 21, 23, 18,
            10, 13, 16, 17, 21, 24, 24, 21,
            14, 18, 19, 21, 22, 20, 20, 20,
        ]

        # Standard JPEG chrominance table (quality ≈ 50)
        chroma_q50 = [
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ]

        transform = albu.Compose([
            PillowJpegCompression(
                luma_tables=[luma_q50, luma_q90],
                chroma_tables=[chroma_q50],
                p=0.5,
            ),
        ])
    """

    def __init__(
        self,
        luma_tables: list,
        chroma_tables: list,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        if not luma_tables:
            raise ValueError("luma_tables must be a non-empty list of quantization tables.")
        if not chroma_tables:
            raise ValueError("chroma_tables must be a non-empty list of quantization tables.")

        for i, table in enumerate(luma_tables):
            if len(table) != 64:
                raise ValueError(
                    f"luma_tables[{i}] has {len(table)} values; each table must have exactly 64."
                )
        for i, table in enumerate(chroma_tables):
            if len(table) != 64:
                raise ValueError(
                    f"chroma_tables[{i}] has {len(table)} values; each table must have exactly 64."
                )

        self.luma_tables = luma_tables
        self.chroma_tables = chroma_tables
        self._last_dct = None
        self._last_qtb = None

    def apply(self, img: np.ndarray, luma_table: list, chroma_table: list, **params) -> np.ndarray:
        """Compress *img* with the given quantization tables, capture DCT/QTB,
        and return the decoded result.

        Args:
            img: Input image as a ``uint8`` HxWxC NumPy array (RGB).
            luma_table: Luminance quantization table (64 integers, zigzag order).
            chroma_table: Chrominance quantization table (64 integers, zigzag order).

        Returns:
            Compressed-then-decoded image as a ``uint8`` NumPy array with the same
            shape as the input.
        """
        pil_img = Image.fromarray(img)

        # Pillow expects qtables as a dict: {component_index: table}
        # Component 0 → luminance (Y), component 1 → chrominance (Cb/Cr).
        qtables = {0: luma_table, 1: chroma_table}

        # --- ONE compression into memory ---
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", qtables=qtables)
        jpeg_bytes = buffer.getvalue()

        # --- Write already-compressed bytes to a temp file (pure I/O) ---
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(jpeg_bytes)
            tmp_path = tmp.name
        try:
            jpg = jpegio.read(tmp_path)
            self._last_dct = jpg.coef_arrays[0].copy()
            self._last_qtb = jpg.quant_tables[0].copy()
        finally:
            os.unlink(tmp_path)

        # --- Decode pixels from the same in-memory buffer ---
        compressed = Image.open(BytesIO(jpeg_bytes))
        compressed.load()
        return np.array(compressed)

    def get_params(self) -> dict:
        """Randomly sample one table from each list."""
        return {
            "luma_table": random.choice(self.luma_tables),
            "chroma_table": random.choice(self.chroma_tables),
        }

    def get_transform_init_args_names(self) -> tuple:
        return ("luma_tables", "chroma_tables")


class JpegCompressionWithDCT(ImageOnlyTransform):
    """JPEG compression augmentation that exposes the DCT coefficients and
    quantization table produced by the **same** compression pass.

    Internally this transform:
    1. Encodes the image to JPEG in a BytesIO buffer (ONE compression).
    2. Writes the already-compressed bytes to a temporary file (pure I/O –
       no second encoding step).
    3. Reads DCT coefficients and the quantization table from that file with
       ``jpegio``.
    4. Decodes the compressed pixels from the same in-memory buffer.

    After each call to ``__call__`` / ``apply`` the results are available as:
    - ``self._last_dct``  – int32 array of quantised DCT coefficients (Y channel)
    - ``self._last_qtb``  – int32 8×8 luminance quantisation table

    Args:
        quality_range (tuple[int, int]): Inclusive ``(min, max)`` quality
            range.  A value is sampled uniformly at random on each call.
            Defaults to ``(30, 100)``.
        p (float): Probability of applying the transform.  Defaults to ``1.0``.
    """

    def __init__(
        self,
        quality_range: tuple = (30, 100),
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.quality_range = quality_range
        self._last_dct = None
        self._last_qtb = None

    def apply(self, img: np.ndarray, quality: int = 75, **params) -> np.ndarray:
        """Compress *img* once, capture DCT/QTB, return decompressed pixels.

        Args:
            img: ``uint8`` HxWxC NumPy array (RGB).
            quality: JPEG quality factor (1–100).

        Returns:
            Compressed-then-decoded image as a ``uint8`` NumPy array with the
            same shape as the input.
        """
        pil_img = Image.fromarray(img)

        # --- ONE compression into memory ---
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality, subsampling=0)
        jpeg_bytes = buffer.getvalue()

        # --- Write already-compressed bytes to a temp file (pure I/O) ---
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(jpeg_bytes)
            tmp_path = tmp.name
        try:
            jpg = jpegio.read(tmp_path)
            self._last_dct = jpg.coef_arrays[0].copy()
            self._last_qtb = jpg.quant_tables[0].copy()
        finally:
            os.unlink(tmp_path)

        # --- Decode pixels from the same in-memory buffer ---
        compressed = Image.open(BytesIO(jpeg_bytes))
        compressed.load()
        return np.array(compressed)

    def get_params(self) -> dict:
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        return {"quality": quality}

    def get_transform_init_args_names(self) -> tuple:
        return ("quality_range",)
