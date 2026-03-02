"""
Post-processing function for ADCD-Net.

Extracts DCT coefficients and quantization tables from JPEG images,
and optionally generates OCR masks for text region identification.
"""

import numpy as np
from ForensicHub.registry import register_postfunc

try:
    from IMDLBenCo.datasets.utils import read_jpeg_from_memory
    HAS_IMDLBENCO = True
except ImportError:
    HAS_IMDLBENCO = False
    print("Warning: IMDLBenCo not found. Using fallback JPEG processing.")


@register_postfunc("adcd_net_post_func")
def adcd_net_post_func(data_dict):
    """
    Post-processing function for ADCD-Net.

    Extracts DCT coefficients and quantization tables from JPEG images.
    The DCT coefficients are clipped to [0, 20] range and the quantization
    table is extracted for adaptive modulation.

    Args:
        data_dict: Dictionary containing:
            - 'image': Raw image tensor/bytes for JPEG processing

    Modifies data_dict in-place to add:
        - 'dct': DCT coefficients (H, W) clipped to [0, 20]
        - 'qt': Quantization table (1, 8, 8)
        - 'ocr_mask': OCR mask (1, H, W) - zeros if OCR not available
    """
    tp_img = data_dict['image']

    if HAS_IMDLBENCO:
        DCT_coef, qtables = __get_jpeg_info(tp_img)
        data_dict['dct'] = np.clip(np.abs(DCT_coef[0]), 0, 20).astype(np.int64)
        data_dict['qt'] = np.expand_dims(qtables[0], axis=0)
    else:
        # Fallback: create dummy DCT and QT if JPEG processing not available
        if hasattr(tp_img, 'shape'):
            h, w = tp_img.shape[:2]
        else:
            h, w = 256, 256
        data_dict['dct'] = np.zeros((h, w), dtype=np.int64)
        data_dict['qt'] = np.ones((1, 8, 8), dtype=np.int64) * 16  # Default QT

    # Add empty OCR mask (can be filled by external OCR processing)
    if 'ocr_mask' not in data_dict:
        if 'mask' in data_dict:
            mask_shape = data_dict['mask'].shape
            if len(mask_shape) == 2:
                data_dict['ocr_mask'] = np.zeros((1, mask_shape[0], mask_shape[1]), dtype=np.float32)
            else:
                data_dict['ocr_mask'] = np.zeros_like(data_dict['mask'], dtype=np.float32)
        else:
            h, w = data_dict['dct'].shape
            data_dict['ocr_mask'] = np.zeros((1, h, w), dtype=np.float32)


def __get_jpeg_info(image_tensor):
    """
    Extract JPEG DCT coefficients and quantization tables.

    Args:
        image_tensor: Image data (raw bytes or tensor)

    Returns:
        DCT_coef: List of DCT coefficient arrays per channel
        qtables: List of quantization tables per channel
    """
    num_channels = 1  # Use only Y channel for document images
    jpeg = read_jpeg_from_memory(image_tensor)

    # Determine subsampling factors
    ci = jpeg.comp_info
    need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]

    if num_channels == 3:
        if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
            need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
        if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
            need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
    else:
        need_scale[0][0] = 2
        need_scale[0][1] = 2

    # Up-sample DCT coefficients to match image size
    DCT_coef = []
    for i in range(num_channels):
        r, c = jpeg.coef_arrays[i].shape
        coef_view = jpeg.coef_arrays[i].reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)

        # Case 1: row scale (O) and col scale (O)
        if need_scale[i][0] == 1 and need_scale[i][1] == 1:
            out_arr = np.zeros((r * 2, c * 2))
            out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

        # Case 2: row scale (O) and col scale (X)
        elif need_scale[i][0] == 1 and need_scale[i][1] == 2:
            out_arr = np.zeros((r * 2, c))
            out_view = out_arr.reshape(r * 2 // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, :, :, :] = coef_view[:, :, :, :]
            out_view[1::2, :, :, :] = coef_view[:, :, :, :]

        # Case 3: row scale (X) and col scale (O)
        elif need_scale[i][0] == 2 and need_scale[i][1] == 1:
            out_arr = np.zeros((r, c * 2))
            out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, ::2, :, :] = coef_view[:, :, :, :]
            out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

        # Case 4: row scale (X) and col scale (X)
        elif need_scale[i][0] == 2 and need_scale[i][1] == 2:
            out_arr = np.zeros((r, c))
            out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, :, :, :] = coef_view[:, :, :, :]

        else:
            raise KeyError("Unexpected scaling factors.")

        DCT_coef.append(out_arr)

    # Quantization tables
    qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(np.float64) for i in range(num_channels)]

    return DCT_coef, qtables