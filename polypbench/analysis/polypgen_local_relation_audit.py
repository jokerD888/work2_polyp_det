"""Primary-test PolypGen audit using existing predictions only; no GPU/model calls."""
import argparse
from collections import defaultdict
import csv
import datetime as dt
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from polypbench.io import read_json,write_json,sha256
from .matching import iou_matrix,xyxy,maximum_coverage
from .local_relation_audit import regions


def run(workspace,output):
    cv2.setNumThreads(1)
    root=Path(workspace).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    old=root/'runs/mssl-polypgen-analysis-20261019';prior=read_json(old/'protocol.json')
    gt=read_json(prior['inputs']['annotations']);image_root=Path(prior['image_root']);groups=defaultdict(list)
    for ann in gt['annotations']:groups[ann['image_id']].append(ann)
    arrays={k:np.load(old/'score_replay'/k/'score_components.npz') for k in ('baseline','mssl')}
    positions={k:{int(i):n for n,i in enumerate(a['image_ids'])} for k,a in arrays.items()}
    images={im['id']:im for im in gt['images']};mask_paths={};inputs=[Path(__file__),Path(__file__).with_name('local_relation_audit.py'),Path(prior['inputs']['annotations']),old/'protocol.json']
    for k in arrays:
        inputs.extend([old/'score_replay'/k/f for f in ('score_components.npz','verification.json')])
        assert set(positions[k])==set(images)
    for im in gt['images']:
        path=image_root/im['file_name'];inputs.append(path)
        if len(groups[im['id']])==1:
            directory=path.parent.parent/('masks_'+im['center'])
            candidates=[directory/(path.stem+'_mask'+suffix) for suffix in ('.jpg','.png','.jpeg')]
            found=[p for p in candidates if p.is_file()]
            if len(found)>1:raise ValueError(f'Ambiguous mask: {path}')
            if found:mask_paths[im['id']]=found[0];inputs.append(found[0])
    protocol={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'cohort_role':'PolypGen primary test benchmark',
        'rules':{'gt':'original provided boxes; masks for descriptive audit only; never replace box labels',
            'inference':'reuse all existing 300 query boxes and original selected100 scores; CPU only',
            'regions':'same fixed geometric core/ring recipe as source audit; not annotation uncertainty',
            'low_contrast':'lowest quartile among all valid single-GT images; outcomes not used to set cutoff',
            'negative_threshold':.5,'coverage':'maximum-cardinality one-to-one',
            'limits':'observational analysis of an already observed test benchmark; no feature or causal claims; no hyperparameter/checkpoint selection'},
        'inputs_sha256':{str(p):sha256(p) for p in inputs}}
    write_json(out/'protocol.json',protocol)
    totals={k:{str(t):{pool:0 for pool in ('all300','selected100','top10')} for t in (.5,.75,.9)} for k in arrays}
    negatives={k:{'images':0,'detections':0} for k in arrays};rows=[];region_cache={};skips=defaultdict(int)
    for im in gt['images']:
        iid=im['id'];anns=groups[iid];g=xyxy([a['bbox'] for a in anns]);model={}
        for k,arr in arrays.items():
            n=positions[k][iid];q=iou_matrix(g,xyxy(arr['all_query_boxes'][n]));s=iou_matrix(g,xyxy(arr['bbox'][n]));order=np.argsort(-arr['final_scores'][n],kind='stable')
            model[k]=(q,s,order)
            for t in (.5,.75,.9):
                dest=totals[k][str(t)];dest['all300']+=maximum_coverage(q,t);dest['selected100']+=maximum_coverage(s,t);dest['top10']+=maximum_coverage(s[:,order[:10]],t)
            if not anns:
                count=int((arr['final_scores'][n]>=.5).sum());negatives[k]['images']+=int(count>0);negatives[k]['detections']+=count
        for j,ann in enumerate(anns):
            row={'image_id':iid,'annotation_id':ann['id'],'file_name':im['file_name'],'center':im['center'],'image_gt_count':len(anns),
                'box_area_ratio':ann['bbox'][2]*ann['bbox'][3]/(im['height']*im['width'])}
            for k,(q,s,order) in model.items():
                row.update({k+'_best300':float(q[j].max()),k+'_best100':float(s[j].max()),k+'_top1_iou':float(s[j,order[0]]),k+'_best300_query':int(q[j].argmax())})
            row['group']='missing' if row['mssl_best300']<.5 else 'coarse' if row['mssl_best300']<.75 else 'ranking' if row['mssl_top1_iou']<.5 else 'covered'
            rows.append(row)
        if len(anns)!=1:continue
        if iid not in mask_paths:skips['missing_mask']+=1;continue
        rgb=np.asarray(Image.open(image_root/im['file_name']).convert('RGB'));mask=np.asarray(Image.open(mask_paths[iid]).convert('L'))>127
        if mask.shape!=rgb.shape[:2]:raise ValueError('Mask/image shape mismatch')
        if not mask.any():skips['empty_mask']+=1;continue
        core,ring=regions(mask)
        if not core.any() or not ring.any():skips['empty_core_or_ring']+=1;continue
        row=rows[-1];lab=cv2.cvtColor(rgb.astype(np.float32)/255.,cv2.COLOR_RGB2Lab)
        row.update(core_pixels=int(core.sum()),ring_pixels=int(ring.sum()),lab_contrast=float(np.linalg.norm(np.median(lab[core],axis=0)-np.median(lab[ring],axis=0))))
        bright=(rgb.max(2)>=245)&(rgb.min(2)>=200);dark=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)<=15
        clean_core=core&~bright&~dark;clean_ring=ring&~bright&~dark
        row['artifact_filtered_contrast']=float(np.linalg.norm(np.median(lab[clean_core],axis=0)-np.median(lab[clean_ring],axis=0))) if clean_core.any() and clean_ring.any() else None
    valid=[r for r in rows if 'lab_contrast' in r];cutoff=float(np.quantile([r['lab_contrast'] for r in valid],.25));sizecut=float(np.quantile([r['box_area_ratio'] for r in valid],.25))
    for row in valid:row.update(low_contrast=row['lab_contrast']<=cutoff,small=row['box_area_ratio']<=sizecut)
    def describe(items):
        return {'n':len(items),'baseline_missing':sum(r['baseline_best300']<.5 for r in items),'mssl_missing':sum(r['mssl_best300']<.5 for r in items),
            'mssl_below075':sum(r['mssl_best300']<.75 for r in items),'median_area_ratio':float(np.median([r['box_area_ratio'] for r in items])) if items else None}
    filtered=[r for r in valid if r['artifact_filtered_contrast'] is not None];fq=float(np.quantile([r['artifact_filtered_contrast'] for r in filtered],.25))
    per_center={}
    for center in sorted({im['center'] for im in images.values()}):
        population=[r for r in rows if r['center']==center];v=[r for r in valid if r['center']==center]
        center_cut=float(np.quantile([r['lab_contrast'] for r in v],.25)) if v else None
        per_center[center]={'all_gt':describe(population),'single_with_regions':describe(v),'global_low_contrast':describe([r for r in v if r['low_contrast']]),
            'within_center_low_contrast':describe([r for r in v if r['lab_contrast']<=center_cut]),'within_center_cutoff':center_cut}
    summary={'cohort_role':'PolypGen primary test benchmark','images':len(images),'gt':len(rows),'negative_images':sum(not groups[i] for i in images),
        'geometry_counts':totals,'negative_score05':negatives,
        'single_gt':describe([r for r in rows if r['image_gt_count']==1]),'multi_gt':describe([r for r in rows if r['image_gt_count']>1]),
        'eligible_region_audit':len(valid),'region_skips':dict(skips),'low_contrast_cutoff':cutoff,'small_cutoff':sizecut,
        'contrast_bins':{str(b):describe([r for r in valid if r['low_contrast']==b]) for b in (True,False)},
        'artifact_filtered_contrast':{'cutoff':fq,'bins':{str(b):describe([r for r in filtered if (r['artifact_filtered_contrast']<=fq)==b]) for b in (True,False)}},
        'size_contrast_cells':{f'low={c},small={s}':describe([r for r in valid if r['low_contrast']==c and r['small']==s]) for c in (True,False) for s in (True,False)},
        'per_center':per_center,'limits':'Handcrafted image contrast, correlated frames and observed test set; no causal or learned-feature separability conclusion. Per-GT maxima are not one-to-one recall.'}
    # Replay must reproduce the old predefined negative-frame diagnostic.
    previous_behavior=read_json(old/'behavior/behavior.json')['cohorts']['pooled']['models']
    for k in arrays:assert negatives[k]['images']==previous_behavior[k]['negative_frames_with_detection_score05']
    write_json(out/'audit.json',summary)
    with (out/'per_gt.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
    selected=[]
    for center in sorted(per_center):
        candidates=sorted([r for r in valid if r['center']==center and r['group']=='missing'],key=lambda r:(r['lab_contrast'],r['image_id']))
        if candidates:selected.append(candidates[0])
    write_json(out/'selected_cases.json',selected)
    canvas=Image.new('RGB',(1000,300*len(selected)),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for index,row in enumerate(selected):
        im=images[row['image_id']];rgb=np.asarray(Image.open(image_root/im['file_name']).convert('RGB'));mask=np.asarray(Image.open(mask_paths[im['id']]).convert('L'))>127
        core,ring=regions(mask);overlay=rgb.copy().astype(float);overlay[core]=.6*overlay[core]+.4*np.array([0,255,0]);overlay[ring]=.6*overlay[ring]+.4*np.array([0,100,255])
        n=positions['mssl'][im['id']];arr=arrays['mssl'];top=int(np.argmax(arr['final_scores'][n]));ann=groups[im['id']][0]
        for column,pixels in enumerate((rgb,overlay.astype(np.uint8))):
            image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
            for box,color in ((ann['bbox'],'yellow'),(arr['bbox'][n,top],'cyan'),(arr['all_query_boxes'][n,row['mssl_best300_query']],'magenta')):
                x,y,w,h=map(float,box);draw.rectangle((x,y,x+w,y+h),outline=color,width=3)
            image.thumbnail((490,250));canvas.paste(image,(column*500+5,index*300+40))
        d.text((5,index*300+2),f"{row['center']} id={row['image_id']} missing best300={row['mssl_best300']:.3f} Lab={row['lab_contrast']:.1f}",font=font,fill='black')
    canvas.save(out/'case_panel.jpg',quality=93)
    for path,digest in protocol['inputs_sha256'].items():
        if sha256(path)!=digest:raise RuntimeError(f'Input changed: {path}')
    write_json(out/'state.json',{'status':'completed','cpu_only':True,'inputs_unchanged':True})
    print({k:v for k,v in summary.items() if k not in ('per_center','size_contrast_cells')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();run(a.workspace,a.output)
