from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import cv2
import torch
import jpegio
import numpy as np
from PIL import Image
from ForensicHub.registry import register_dataset
from ForensicHub.core.base_dataset import BaseDataset
import magic

@register_dataset("DocDataset")
class DocDataset(BaseDataset):
    def __init__(self, path, train=True, crop_size=512, jpeg=True, dct_path=None, crop_test=False, suffix_img='.jpg', suffix_mask='.png',get_dct_qtb:bool=False,**kwargs):
        self.jpeg = True # must use jpeg augmentation
        self.train = train # train or test
        self.dct_path = dct_path # return dct or not
        self.crop_size = crop_size # model input size
        self.crop_test = False
        self.path = path
        if isinstance(suffix_img,list):
            suffix_img = tuple(suffix_img)
        self.suffix_img = suffix_img
        self.suffix_mask = suffix_mask
        self.get_dct_qtb = get_dct_qtb
        # if dct_path:
            # # other_files/qt_table_ori.pk
            # with open(dct_path, 'rb') as f:
            #     self.qtables = pickle.load(f)
        super().__init__(path=path, **kwargs)
        print(path, self.__len__, 'train:', train)


    def _init_dataset_path(self):
        images_dir = os.path.join(self.path, "images")
        masks_dir = os.path.join(self.path, "masks")

        # 1) Liste des candidats
        filenames = [
            fn for fn in os.listdir(images_dir)
            if fn.lower().endswith(self.suffix_img)
        ]

        def process_one(filename):
            image_path = os.path.join(images_dir, filename)

            # Lire la taille sans charger tous les pixels (header only en pratique)
            try:
                with Image.open(image_path) as img:
                    w, h = img.size
            except Exception:
                # fichier corrompu / illisible
                return None

            if w < 512 or h < 512:
                return None

            mask_name = filename[:-len(filename.split(".")[-1])-1] + self.suffix_mask
            mask_path = os.path.join(masks_dir, mask_name)
            if not os.path.isfile(mask_path):
                mask_path = None

            return (image_path, mask_path)

        # 2) Parallélisation
        # max_workers: threads = ok pour I/O. Tu peux aussi mettre min(32, (os.cpu_count() or 1) * 4)
        max_workers = min(32, (os.cpu_count() or 1) * 4)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = ex.map(process_one, filenames, chunksize=128)

        # 3) Filtrer les None
        self.images = [r for r in results if r is not None]

    def __len__(self):
        return len(self.images)

    def _find_jpeg_dct_transform(self):
        """Return the first transform inside common_transform that exposes
        ``_last_dct`` and ``_last_qtb`` (i.e. JpegCompressionWithDCT or
        PillowJpegCompression), or None if none is found."""
        if self.common_transform is None:
            return None
        for t in getattr(self.common_transform, 'transforms', []):
            if hasattr(t, '_last_dct') and hasattr(t, '_last_qtb'):
                return t
        return None

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]
        image = Image.open(img_path)
        h, w = image.size
        img_real_path = os.path.realpath(img_path)
        mime = magic.from_file(img_real_path, mime=True)
        if mask_path:
            mask = np.clip(cv2.imread(mask_path, 0), 0, 1)
        else:
            mask = np.zeros((w, h), dtype=np.uint8)

        # --- Test mode: read DCT/QTB from the original file (no augmentation) ---
        if not self.train and self.get_dct_qtb:
            if mime == "image/jpeg":
                jpg = jpegio.read(img_path)
                dct = jpg.coef_arrays[0].copy()
                qtb = jpg.quant_tables[0].copy()
            else:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                    tmp_path = tmp.name
                    image.save(tmp_path, format="JPEG", quality=100, subsampling=0)
                    image = Image.open(tmp_path)
                    jpg = jpegio.read(tmp_path)
                    dct = jpg.coef_arrays[0].copy()
                    qtb = jpg.quant_tables[0].copy()

        image = np.array(image)
        if self.common_transform:
            output = self.common_transform(image=image, mask=mask)
            image = output['image']
            mask = output['mask']

        # --- Train mode: get DCT/QTB from the JpegCompressionWithDCT that ran inside
        #     common_transform — same single compression pass, no second encoding. ---
        if self.train and self.get_dct_qtb:
            jpeg_t = self._find_jpeg_dct_transform()
            if jpeg_t is not None and jpeg_t._last_dct is not None:
                dct = jpeg_t._last_dct
                qtb = jpeg_t._last_qtb
            else:
                # Fallback if the transform was not found or did not fire (p < 1)
                if mime == "image/jpeg":
                    jpg = jpegio.read(img_path)
                    dct = jpg.coef_arrays[0].copy()
                    qtb = jpg.quant_tables[0].copy()
                else:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                        tmp_path = tmp.name
                        Image.fromarray(image).save(tmp_path, format="JPEG", quality=100, subsampling=0)
                        jpg = jpegio.read(tmp_path)
                        dct = jpg.coef_arrays[0].copy()
                        qtb = jpg.quant_tables[0].copy()

        mask = torch.LongTensor(mask)
        mask = mask.unsqueeze(0)
        label = (mask.sum(dim=(0, 1, 2)) != 0).long()
        if self.post_transform:
            image = self.post_transform(image=image)['image']
        if self.get_dct_qtb:
            return {'image': image, 'mask': mask, 'label': label, 'dct': np.clip(np.abs(dct), 0, 20), 'qt': qtb}
        else:
            return {'image': image, 'mask': mask, 'label': label}

if __name__=='__main__':
    data_names = (('/mnt/data0/public_datasets/Doc/DocTamperV1/DocTamperV1-TrainingSet', False), ('/mnt/data0/public_datasets/Doc/DocTamperV1/DocTamperV1-TrainingSet', True))
    for v in data_names:
        data = DocDataset(path=v[0], train=v[1])
        for i in range(10):
            item = data.__getitem__(0)
            img = item['image']
            mask = item['mask']
        import pdb;pdb.set_trace()
            # if use_dct:
            #     dct = item['dct']
            #     qtb = item['qtb']
            #     print(data_names, i, img.shape, mask.shape, dct.shape, qtb.shape)
            # else:
            #     print(data_names, i, img.shape, mask.shape)
             
            
