"""Inspect semantic-mask/box support without treating it as instance annotation."""
import argparse
from collections import defaultdict
import datetime as dt
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from polypbench.io import read_json,write_json,sha256
from .matching import iou_matrix,xyxy


def run(workspace,output):
    cv2.setNumThreads(1)
    root=Path(workspace).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    audit=root/'runs/local-relation-feasibility-20261020';p=read_json(audit/'protocol.json');prior=read_json(audit/'audit.json')
    gt=read_json(p['annotations']);groups=defaultdict(list)
    for ann in gt['annotations']:groups[ann['image_id']].append(ann)
    missing={r['annotation_id'] for r in prior['all_missing_gt']};rows=[];panels=[]
    inputs=[Path(__file__),Path(p['annotations']),audit/'audit.json']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for im in gt['images']:
        anns=groups[im['id']];maskpath=Path(p['raw_mask_root'])/Path(im['file_name']).name
        inputs.append(maskpath)
        mask=np.asarray(Image.open(maskpath).convert('L'))>127;h,w=mask.shape
        ys,xs=np.nonzero(mask);tight=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)] if len(xs) else [0,0,0,0]
        labels_n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        components=sorted(stats[1:,cv2.CC_STAT_AREA].tolist(),reverse=True)
        for ann in anns:
            x,y,bw,bh=ann['bbox'];x0,y0=max(0,int(np.floor(x))),max(0,int(np.floor(y)))
            x1,y1=min(w,int(np.ceil(x+bw))),min(h,int(np.ceil(y+bh)))
            count=int(mask[y0:y1,x0:x1].sum());area=max(0,x1-x0)*max(0,y1-y0)
            rows.append({'image_id':im['id'],'annotation_id':ann['id'],'file_name':im['file_name'],
                'image_gt_count':len(anns),'mask_pixels_in_box':count,'box_mask_fraction':count/area if area else None,
                'fraction_of_semantic_fg_in_box':count/max(1,int(mask.sum())),
                'bbox_iou_with_semantic_tight_box':float(iou_matrix(xyxy([ann['bbox']]),xyxy([tight]))[0,0]),
                'mssl_all300_missing_at05':ann['id'] in missing,
                'semantic_component_areas':components,'box':ann['bbox']})
        if len(anns)>1:
            imagepath=Path(p['image_root'])/im['file_name'];inputs.append(imagepath)
            rgb=np.asarray(Image.open(imagepath).convert('RGB'));overlay=rgb.copy().astype(float)
            overlay[mask]=.65*overlay[mask]+.35*np.array([0,255,0]);image=Image.fromarray(overlay.astype(np.uint8));d=ImageDraw.Draw(image)
            for index,ann in enumerate(anns):
                x,y,bw,bh=map(float,ann['bbox']);color=('yellow','cyan','magenta')[index%3]
                d.rectangle((x,y,x+bw,y+bh),outline=color,width=3)
            image.thumbnail((390,270));tile=Image.new('RGB',(400,310),'white');tile.paste(image,(5,35));d=ImageDraw.Draw(tile)
            ids=','.join(str(a['id'])+('*' if a['id'] in missing else '') for a in anns)
            d.text((5,2),f"image {im['id']} GT {ids} (*missing)",font=font,fill='black');panels.append(tile)
    summary={'images':len(gt['images']),'gt':len(rows),'single_gt':sum(r['image_gt_count']==1 for r in rows),
        'multi_gt':sum(r['image_gt_count']>1 for r in rows),'all300_missing':len(missing),
        'missing_on_multi_gt':sum(r['mssl_all300_missing_at05'] and r['image_gt_count']>1 for r in rows),
        'missing_with_no_semantic_mask_pixels_in_box':sum(r['mssl_all300_missing_at05'] and r['mask_pixels_in_box']==0 for r in rows),
        'missing_with_box_mask_fraction_below005':sum(r['mssl_all300_missing_at05'] and r['box_mask_fraction']<.05 for r in rows),
        'per_gt':rows,'limits':'Raw semantic masks do not establish per-instance identity or normal-mucosa labels. No correction or relabeling performed.'}
    write_json(out/'audit.json',summary)
    canvas=Image.new('RGB',(800,310*((len(panels)+1)//2)),'white')
    for i,tile in enumerate(panels):canvas.paste(tile,((i%2)*400,(i//2)*310))
    canvas.save(out/'multi_gt_panel.jpg',quality=93)
    write_json(out/'protocol.json',{'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'inputs_sha256':{str(path):sha256(path) for path in inputs},'cpu_only':True,'no_annotation_mutations':True})
    print({k:v for k,v in summary.items() if k!='per_gt'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();run(a.workspace,a.output)
