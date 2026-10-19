"""Read-only PolypGen still-frame conversion and source-overlap audit.

Primary GT uses supplied text boxes verbatim: their bundled generator uses
regionprops' zero-based, exclusive-max coordinates despite the VOC name.
Missing box files are unscored, never silently converted to negative images.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .io import read_json, sha256, write_json
from .data import validate_coco


def parse_boxes(text, width, height):
    result=[]
    for line in text.splitlines():
        if not line.strip():
            continue
        fields=line.split()
        if len(fields)!=5 or fields[0]!="polyp":
            raise ValueError(f"Unexpected box row: {line}")
        x1,y1,x2,y2=map(float,fields[1:])
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):
            raise ValueError(f"Out-of-bounds/degenerate box: {line}, image {width}x{height}")
        box=[x1,y1,x2-x1,y2-y1]
        if box in result:
            raise ValueError("Duplicate supplied GT box")
        result.append(box)
    return result


def fingerprint_image(path):
    path=Path(path)
    encoded=sha256(path)
    with Image.open(path) as image:
        rgb=image.convert("RGB")
        size=rgb.size
        decoded=hashlib.sha256(str(size).encode()+rgb.tobytes()).hexdigest()
        gray=np.asarray(rgb.convert("L").resize((17,16),Image.Resampling.LANCZOS))
        bits=(gray[:,1:]>gray[:,:-1]).reshape(-1)
        dhash=int.from_bytes(np.packbits(bits).tobytes(),"big")
    return {"path":str(path),"width":size[0],"height":size[1],
            "sha256":encoded,"rgb_sha256":decoded,"dhash256":f"{dhash:064x}"}


def prepare(source, output, source_annotations):
    root=Path(source).resolve()
    if not (root/"data_C1").is_dir():
        roots=[p for p in root.iterdir() if p.is_dir() and (p/"data_C1").is_dir()]
        if len(roots)!=1:
            raise ValueError("Cannot identify the PolypGen data root")
        root=roots[0]
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    files=sorted(p for i in range(1,7) for p in
                 (root/f"data_C{i}"/f"images_C{i}").glob("*.jpg"))
    if not files:
        raise ValueError("No center-split still images")
    source_files=[]
    for annotation in source_annotations:
        annotation=Path(annotation).resolve()
        data=read_json(annotation)
        image_root=annotation.parent/annotation.stem
        source_files += [image_root/x["file_name"] for x in data["images"]]
    print(f"Hashing {len(files)} target and {len(source_files)} source images",flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        target_meta=list(pool.map(fingerprint_image,files))
        source_meta=list(pool.map(fingerprint_image,source_files))
    source_encoded=defaultdict(list);source_decoded=defaultdict(list)
    for item in source_meta:
        source_encoded[item["sha256"]].append(item["path"])
        source_decoded[item["rgb_sha256"]].append(item["path"])
    all_images=[];known_images=[];annotations=[];exclusions=[];near=[]
    entries=[];exact_overlaps=[];internal=defaultdict(list)
    ann_id=1
    source_hashes=[int(x["dhash256"],16) for x in source_meta]
    for image_id,(file,meta) in enumerate(zip(files,target_meta),1):
        center=file.parent.name.removeprefix("images_")
        image={"id":image_id,"file_name":file.relative_to(root).as_posix(),
               "width":meta["width"],"height":meta["height"],"center":center}
        all_images.append(image)
        matches=sorted(set(source_encoded[meta["sha256"]]+source_decoded[meta["rgb_sha256"]]))
        internal[meta["rgb_sha256"]].append(image_id)
        dh=int(meta["dhash256"],16)
        distances=[(dh^x).bit_count() for x in source_hashes]
        if distances:
            best=min(distances)
            if best<=12:
                for j,distance in enumerate(distances):
                    if distance==best:
                        near.append({"target":image["file_name"],"source":source_meta[j]["path"],
                                     "dhash256_hamming":distance})
        box_dir=file.parent.parent/f"bbox_{center}"
        candidates=[box_dir/(file.stem+suffix+".txt") for suffix in ("","_mask")]
        existing=[p for p in candidates if p.is_file()]
        if len(existing)>1:
            raise ValueError(f"Ambiguous supplied box file: {file}")
        entry={**meta,"image_id":image_id,"center":center,"relative_path":image["file_name"],
               "bbox_path":str(existing[0]) if existing else None,
               "bbox_sha256":sha256(existing[0]) if existing else None}
        if matches:
            exact_overlaps.append({"target":image["file_name"],"source":matches})
        if not existing or matches:
            exclusions.append({"image_id":image_id,"file_name":image["file_name"],"center":center,
                "reason":"missing_supplied_bbox" if not existing else "exact_source_overlap"})
        else:
            boxes=parse_boxes(existing[0].read_text(),meta["width"],meta["height"])
            known_images.append(image)
            for box in boxes:
                annotations.append({"id":ann_id,"image_id":image_id,"category_id":1,
                                    "bbox":box,"area":box[2]*box[3],"iscrowd":0})
                ann_id+=1
            entry["box_count"]=len(boxes)
        entries.append(entry)
    categories=[{"id":1,"name":"polyp"}]
    base={"info":{"description":"PolypGen C1-C6 supplied-box zero-shot protocol"},"categories":categories}
    write_json(out/"all_stills.images.json",dict(base,images=all_images,annotations=[]))
    write_json(out/"provided_boxes.coco.json",dict(base,images=known_images,annotations=annotations))
    centers={}
    for i in range(1,7):
        center=f"C{i}"
        images=[x for x in known_images if x["center"]==center];ids={x["id"] for x in images}
        anns=[x for x in annotations if x["image_id"] in ids]
        write_json(out/f"{center}.coco.json",dict(base,images=images,annotations=anns))
        positive={x["image_id"] for x in anns}
        centers[center]={"images":len(images),"boxes":len(anns),"negative_images":len(ids-positive)}
    write_json(out/"image_manifest.json",entries)
    write_json(out/"source_image_manifest.json",source_meta)
    duplicate_groups=[ids for ids in internal.values() if len(ids)>1]
    ambiguous_ids=set()
    for ids in duplicate_groups:
        # Quarantine every copy for sensitivity analysis: choosing one of
        # conflicting annotations would require new manual ground truth.
        ambiguous_ids.update(ids)
    clean_images=[x for x in known_images if x["id"] not in ambiguous_ids]
    clean_ids={x["id"] for x in clean_images}
    write_json(out/"without_internal_duplicates.coco.json",dict(base,images=clean_images,
        annotations=[x for x in annotations if x["image_id"] in clean_ids]))
    audit={"source_root":str(root),"all_stills":len(all_images),"scored_images":len(known_images),
           "boxes":len(annotations),"centers":centers,"exclusions":exclusions,
           "source_annotations":[{"path":str(Path(p).resolve()),"sha256":sha256(p)} for p in source_annotations],
           "exact_source_overlaps":exact_overlaps,
           "internal_exact_pixel_duplicates":duplicate_groups,
           "duplicate_sensitivity_images":len(clean_images),
           "near_source_candidates_hamming_le_12":near,
           "duplicate_audit_limits":"Exact encoded/decoded images and 256-bit dHash screen; not proof of patient/video disjointness.",
           "annotation_policy":"Provided zero-based exclusive-max boxes, including empty-file negatives. Missing files unscored. No mask-derived GT.",
           "scope":"Only center-split still frames; concatenated copies, overlays and video sequences not pooled."}
    write_json(out/"audit.json",audit)
    shutil_files=("readme.md","codes/extract_PolypBoxes.py","codes/convert2vocFromMask.py")
    for name in shutil_files:
        path=root/name
        target=out/"source_notes"/Path(name).name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    validate_coco(out/"provided_boxes.coco.json",root,check_images=False)
    print(json.dumps({k:audit[k] for k in ("all_stills","scored_images","boxes","centers")},indent=2),flush=True)
    print(f"Exact source overlaps: {len(exact_overlaps)}, near candidates: {len(near)}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--source-annotations",nargs="+",required=True)
    args=parser.parse_args()
    prepare(args.source,args.output,args.source_annotations)


if __name__=="__main__":
    main()
