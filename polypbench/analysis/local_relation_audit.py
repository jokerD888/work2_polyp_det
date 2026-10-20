"""CPU-only, read-only failure/region audit; no learned-feature or causal claims."""
import argparse
from collections import defaultdict
import csv
import datetime as dt
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from polypbench.io import read_json, write_json, sha256
from .matching import iou_matrix, maximum_coverage, xyxy


RULES = {
    'cohort': 'all existing Kvasir Val predictions; raw semantic mask only for original single-GT images',
    'input_model': 'none; reuse frozen native selected100 and all300 outputs; CPU only',
    'core': 'inside raw mask, distance to mask exterior > max(1,ceil(.05*sqrt(mask_area))) pixels',
    'ring': 'outside raw mask, distance in (max(1,ceil(.05*sqrt(area))), max(2,ceil(.20*sqrt(area)))] pixels',
    'boundary_interpretation': 'geometry proxy, not an estimate of annotation or epistemic uncertainty',
    'artifact_proxies': 'highlight-like maxRGB>=245 and minRGB>=200; dark-like grayscale<=15; descriptive only',
    'contrast': 'Euclidean distance of median CIELAB core/ring colors; not learned-feature separability',
    'low_contrast_bin': 'lowest quartile of all valid eligible images, determined without detection outcomes',
    'small_bin': 'lowest area-ratio quartile in same eligible cohort; descriptive confounding check',
    'failure_groups': 'all300 best IoU <.5: missing; [.5,.75): coarse; >=.75 but native top1<.5: ranking; otherwise covered',
    'coverage': 'maximum-cardinality one-to-one for all-GT counts; per-GT maxima are marginal diagnostics only',
    'limits': 'source Val used for development; zero negative images; no new training, no target data; no causal attribution or clinical artifact labels',
}


def regions(mask):
    area=int(mask.sum())
    inner=max(1,int(np.ceil(.05*np.sqrt(area))));outer=max(2,int(np.ceil(.20*np.sqrt(area))))
    # Pad explicitly so a foreground region touching the image boundary still
    # has a well-defined distance to the outside of the valid image rectangle.
    inside=cv2.distanceTransform(np.pad(mask.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    outside=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return mask & (inside>inner), (~mask)&(outside>inner)&(outside<=outer)


def group(row):
    q=row['mssl_best300']
    return 'missing' if q<.5 else 'coarse' if q<.75 else 'ranking' if row['mssl_top1_iou']<.5 else 'covered'


def run(workspace, output):
    cv2.setNumThreads(1)
    root=Path(workspace).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    previous=root/'runs/mssl-quality-source-20261020';old=read_json(previous/'protocol.json')
    gt=read_json(old['annotations']);images={im['id']:im for im in gt['images']};annotations=defaultdict(list)
    for ann in gt['annotations']:annotations[ann['image_id']].append(ann)
    trained=Path(old['training']['mssl']['run_root']);e=read_json(trained/'experiment.json')
    arrays={label:np.load(previous/'score_replay'/label/'score_components.npz') for label in ('baseline','mssl')}
    files=[Path(old['annotations']),trained/'experiment.json',previous/'protocol.json',Path(__file__)]
    files += [previous/'score_replay'/label/name for label in arrays for name in ('score_components.npz','verification.json')]
    for im in images.values():
        files.append(Path(old['image_root'])/im['file_name'])
        if len(annotations[im['id']])==1:files.append(Path(e['dataset']['raw_mask_root'])/Path(im['file_name']).name)
    protocol={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'rules':RULES,
        'annotations':old['annotations'],'image_root':old['image_root'],'raw_mask_root':e['dataset']['raw_mask_root'],
        'inputs_sha256':{str(p):sha256(p) for p in files},
        'reference_checkpoint_sha256':{k:read_json(previous/'score_replay'/k/'verification.json')['checkpoint_sha256'] for k in arrays}}
    write_json(out/'protocol.json',protocol)
    rows=[];coverage={};positions={k:{int(i):n for n,i in enumerate(a['image_ids'])} for k,a in arrays.items()}
    totals={k:{str(t):{'all300':0,'selected100':0,'top10':0} for t in (.5,.75,.9)} for k in arrays}
    for iid,im in sorted(images.items()):
        anns=annotations[iid];gboxes=xyxy([a['bbox'] for a in anns]);model_rows={}
        for label,arr in arrays.items():
            n=positions[label][iid];q=iou_matrix(gboxes,xyxy(arr['all_query_boxes'][n]));s=iou_matrix(gboxes,xyxy(arr['bbox'][n]))
            order=np.argsort(-arr['final_scores'][n],kind='stable');model_rows[label]=(q,s,order)
            for t in (.5,.75,.9):
                dest=totals[label][str(t)];dest['all300']+=maximum_coverage(q,t);dest['selected100']+=maximum_coverage(s,t)
                dest['top10']+=maximum_coverage(s[:,order[:10]],t)
        for j,ann in enumerate(anns):
            row={'annotation_id':ann['id'],'image_id':iid,'file_name':im['file_name'],'image_gt_count':len(anns),
                 'box_area_ratio':ann['bbox'][2]*ann['bbox'][3]/(im['height']*im['width'])}
            for label,(q,s,order) in model_rows.items():
                row.update({label+'_best300':float(q[j].max()),label+'_best100':float(s[j].max()),
                    label+'_top1_iou':float(s[j,order[0]]),label+'_best300_query':int(q[j].argmax())})
            row['group']=group(row);rows.append(row)
        if len(anns)!=1:continue
        row=rows[-1];rgb=np.asarray(Image.open(Path(old['image_root'])/im['file_name']).convert('RGB'))
        mask=np.asarray(Image.open(Path(e['dataset']['raw_mask_root'])/Path(im['file_name']).name).convert('L'))>127
        if mask.shape!=rgb.shape[:2]:raise ValueError('Raw mask size mismatch')
        core,ring=regions(mask);row.update(core_pixels=int(core.sum()),ring_pixels=int(ring.sum()),mask_pixels=int(mask.sum()))
        if not core.any() or not ring.any():continue
        lab=cv2.cvtColor(rgb.astype(np.float32)/255.,cv2.COLOR_RGB2Lab)
        row['lab_contrast']=float(np.linalg.norm(np.median(lab[core],axis=0)-np.median(lab[ring],axis=0)))
        highlight=(rgb.max(2)>=245)&(rgb.min(2)>=200);dark=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)<=15
        for name,region in (('core',core),('ring',ring)):
            row[name+'_highlight_like_fraction']=float(highlight[region].mean());row[name+'_dark_like_fraction']=float(dark[region].mean())
        clean_core=core&~highlight&~dark;clean_ring=ring&~highlight&~dark
        row['artifact_filtered_contrast']=float(np.linalg.norm(np.median(lab[clean_core],axis=0)-np.median(lab[clean_ring],axis=0))) if clean_core.any() and clean_ring.any() else None
    valid=[r for r in rows if 'lab_contrast' in r]
    cq=float(np.quantile([r['lab_contrast'] for r in valid],.25));sq=float(np.quantile([r['box_area_ratio'] for r in valid],.25))
    for r in valid:r.update(low_contrast=r['lab_contrast']<=cq,small=r['box_area_ratio']<=sq)
    def describe(items):
        return {'n':len(items),'mssl_missing':sum(r['mssl_best300']<.5 for r in items),
            'baseline_missing':sum(r['baseline_best300']<.5 for r in items),
            'mssl_below075':sum(r['mssl_best300']<.75 for r in items),
            'median_area_ratio':float(np.median([r['box_area_ratio'] for r in items])) if items else None,
            'median_lab_contrast':float(np.median([r['lab_contrast'] for r in items])) if items else None}
    summary={'images':len(images),'gt':len(rows),'negative_images':sum(not annotations[i] for i in images),
        'geometry_counts':totals,'eligible_region_audit':len(valid),'ineligible_multi_gt':sum(r['image_gt_count']>1 for r in rows),
        'quartiles':{'lab_contrast':cq,'area_ratio':sq},'groups':{g:describe([r for r in valid if r['group']==g]) for g in ('missing','coarse','ranking','covered')},
        'contrast_bins':{str(b):describe([r for r in valid if r['low_contrast']==b]) for b in (True,False)},
        'size_contrast_cells':{f'low_contrast={c},small={s}':describe([r for r in valid if r['low_contrast']==c and r['small']==s]) for c in (True,False) for s in (True,False)},
        'all_missing_gt':[r for r in rows if r['mssl_best300']<.5],
        'interpretation':'Handcrafted image contrast and geometry are descriptive, not neural separability or causal evidence for contrastive training.'}
    write_json(out/'audit.json',summary)
    with (out/'per_gt.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
    # Deterministic examples from every failure class; include low-contrast
    # successfully covered controls rather than showing only persuasive errors.
    selected=[]
    for name,number in (('missing',3),('coarse',1),('ranking',1),('covered',1)):
        candidates=sorted([r for r in valid if r['group']==name],key=lambda r:(r['mssl_best300'] if name!='covered' else r['lab_contrast'],r['image_id']))
        selected.extend(candidates[:number])
    write_json(out/'selected_cases.json',selected)
    canvas=Image.new('RGB',(1000,300*len(selected)),(245,245,245));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for panel,row in enumerate(selected):
        im=images[row['image_id']];rgb=np.asarray(Image.open(Path(old['image_root'])/im['file_name']).convert('RGB'))
        mask=np.asarray(Image.open(Path(e['dataset']['raw_mask_root'])/Path(im['file_name']).name).convert('L'))>127;core,ring=regions(mask)
        overlay=rgb.copy().astype(float);overlay[core]=.6*overlay[core]+.4*np.array([0,255,0]);overlay[ring]=.6*overlay[ring]+.4*np.array([0,100,255])
        n=positions['mssl'][row['image_id']];arr=arrays['mssl'];top=int(np.argmax(arr['final_scores'][n]));ann=annotations[row['image_id']][0]
        for col,pixels in enumerate((rgb,overlay.astype(np.uint8))):
            image=Image.fromarray(pixels);d=ImageDraw.Draw(image)
            for box,color in ((ann['bbox'],'yellow'),(arr['bbox'][n,top],'cyan'),(arr['all_query_boxes'][n,row['mssl_best300_query']],'magenta')):
                x,y,w,h=map(float,box);d.rectangle((x,y,x+w,y+h),outline=color,width=3)
            image.thumbnail((490,250));canvas.paste(image,(col*500+5,panel*300+45))
        draw.text((5,panel*300+2),f"id={row['image_id']} {row['group']} best300={row['mssl_best300']:.3f} top1={row['mssl_top1_iou']:.3f} Lab={row['lab_contrast']:.1f}",font=font,fill='black')
        draw.text((505,panel*300+24),'core green / ring blue; GT yellow, top1 cyan, best300 magenta',font=font,fill='black')
    canvas.save(out/'case_panel.jpg',quality=92)
    for path,digest in protocol['inputs_sha256'].items():
        if sha256(path)!=digest:raise RuntimeError(f'Input changed during audit: {path}')
    write_json(out/'state.json',{'status':'completed','cpu_only':True,'inputs_unchanged':True})
    print(json.dumps({k:v for k,v in summary.items() if k!='all_missing_gt'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();run(a.workspace,a.output)
