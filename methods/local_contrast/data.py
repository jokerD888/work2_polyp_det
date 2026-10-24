from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from methods.mssl.data import load_records as mssl_records, MSSLMapper


def load_records(annotations, image_root, dataset_name, raw_mask_root):
    records = mssl_records(annotations, image_root, dataset_name, raw_mask_root)
    for record in records:
        # Full semantic union is only a background exclusion for multi-box data.
        # It never becomes an instance label or replaces native multi-box masks.
        with Image.open(Path(raw_mask_root)/Path(record['file_name']).name) as image:
            mask = np.asarray(image.convert('L')) > 127
        if mask.shape != (record['height'], record['width']):
            raise ValueError('Semantic mask/image dimensions differ')
        record['local_semantic_rle'] = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
    return records


class LocalMapper(MSSLMapper):
    """The common mapper transforms the optional union with the same LSJ object."""
