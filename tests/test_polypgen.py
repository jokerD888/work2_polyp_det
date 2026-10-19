import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from polypbench.polypgen import parse_boxes,prepare
from polypbench.io import sha256


class PolypGenTests(unittest.TestCase):
    def test_zero_based_exclusive_max_and_multiple_objects(self):
        self.assertEqual(parse_boxes('polyp 0 0 1 1\npolyp 2 3 8 9',10,10),
                         [[0.,0.,1.,1.],[2.,3.,6.,6.]])

    def test_empty_file_is_negative_but_bad_rows_fail(self):
        self.assertEqual(parse_boxes('\n\n',10,10),[])
        for text in ('polyp 1 1 1 2','polyp -1 0 2 2','polyp 1 1 11 12',
                     'polyp 0 0 1 1\npolyp 0 0 1 1','other 0 0 1 1'):
            with self.assertRaises(ValueError):parse_boxes(text,10,10)

    def test_missing_boxes_are_unscored_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'raw';root.mkdir()
            (root/'codes').mkdir()
            for name in ('readme.md','codes/extract_PolypBoxes.py','codes/convert2vocFromMask.py'):
                (root/name).write_text('synthetic provenance')
            for i in range(1,7):
                d=root/f'data_C{i}'
                images=d/f'images_C{i}';images.mkdir(parents=True)
                boxes=d/f'bbox_C{i}';boxes.mkdir()
                Image.new('RGB',(10,10),(i*30,20,80)).save(images/f'c{i}.jpg')
                if i!=3:
                    (boxes/f'c{i}.txt').write_text('' if i==1 else 'polyp 1 2 8 9')
            source=Path(temp)/'source';(source/'train').mkdir(parents=True)
            Image.new('RGB',(8,8),(10,200,5)).save(source/'train/s.jpg')
            annotation=source/'train.json'
            annotation.write_text(json.dumps({'images':[{'file_name':'s.jpg'}]}))
            before={str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
            out=Path(temp)/'converted'
            with contextlib.redirect_stdout(io.StringIO()):prepare(root,out,[annotation])
            audit=json.loads((out/'audit.json').read_text())
            self.assertEqual(audit['all_stills'],6)
            self.assertEqual(audit['scored_images'],5)
            self.assertEqual(audit['centers']['C1']['negative_images'],1)
            self.assertEqual(audit['exclusions'][0]['reason'],'missing_supplied_bbox')
            self.assertEqual(before,{str(p):sha256(p) for p in root.rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main()
