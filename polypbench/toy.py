import math
from pathlib import Path

from PIL import Image, ImageDraw

from .io import write_json


def create_toy(root):
    """Synthetic polygon masks for plumbing checks, never a research benchmark."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        directory = root / split
        directory.mkdir()
        images, annotations = [], []
        for index in range(count):
            image = Image.new("RGB", (256, 256), (35 + index * 5, 25, 30))
            # Different shapes between splits; category ID 7 catches erroneous 0/1 mapping.
            cx, cy = 95 + index * 10 + len(split), 115
            polygon = [coordinate for k in range(24) for coordinate in (cx + 30 * math.cos(k * math.pi / 12), cy + 25 * math.sin(k * math.pi / 12))]
            ImageDraw.Draw(image).polygon(list(zip(polygon[::2], polygon[1::2])), fill=(190, 90 + index * 5, 95))
            filename = f"{split}_{index}.png"
            image.save(directory / filename)
            images.append({"id": index + 1, "file_name": filename, "width": 256, "height": 256})
            annotations.append({"id": index + 1, "image_id": index + 1, "category_id": 7, "bbox": [cx - 30, cy - 25, 60, 50], "area": 30 * 25 * 12 * math.sin(math.pi / 12), "segmentation": [polygon], "iscrowd": 0})
        write_json(root / f"{split}.json", {"info": {"description": "SYNTHETIC SMOKE TEST ONLY"}, "images": images, "annotations": annotations, "categories": [{"id": 7, "name": "polyp"}]})
