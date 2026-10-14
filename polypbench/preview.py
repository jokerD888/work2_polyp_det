from pathlib import Path

from PIL import Image, ImageDraw

from .io import read_json


def preview(annotations, predictions, image_root, output, limit=8, max_predictions=5):
    ground_truth, rows = read_json(annotations), read_json(predictions)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    root = Path(image_root).resolve()
    saved = []
    for image in ground_truth["images"][:limit]:
        file = (root / image["file_name"]).resolve()
        if not file.is_relative_to(root):
            raise ValueError("Preview image path escapes image root")
        with Image.open(file) as source:
            canvas = source.convert("RGB")
        canvas.thumbnail((1000, 800))
        sx, sy = canvas.width / image["width"], canvas.height / image["height"]
        draw = ImageDraw.Draw(canvas)
        for ann in ground_truth["annotations"]:
            if ann["image_id"] == image["id"]:
                x, y, w, h = ann["bbox"]
                draw.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], outline="lime", width=3)
        predicted = sorted((r for r in rows if r["image_id"] == image["id"]), key=lambda r: r["score"], reverse=True)[:max_predictions]
        for row in predicted:
            x, y, w, h = row["bbox"]
            draw.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], outline="orange", width=2)
            draw.text((x * sx + 2, y * sy + 2), f"{row['score']:.3f}", fill="orange", stroke_width=1, stroke_fill="black")
        result = Image.new("RGB", (canvas.width, canvas.height + 35), "black")
        result.paste(canvas, (0, 35))
        ImageDraw.Draw(result).text((8, 10), f"Green: GT | Orange: top {max_predictions} predictions | {image['file_name']}", fill="white")
        name = target / f"{image['id']}.png"
        result.save(name)
        saved.append(str(name))
    return saved
