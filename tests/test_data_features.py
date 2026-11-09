import unittest
import tempfile
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from gradation.data import validate_manifest, assert_independent, assign_slab_splits, physical_windows, resize_labels
from gradation.feature47 import measure, HEADERS, read_labels, build_features
from gradation.database import build_database, align_tables, TARGETS

class DataFeaturesTests(unittest.TestCase):
    def test_slab_and_independence(self):
        rows = []
        for slab in range(40):
            for patch in range(128):
                rows.append(dict(image_id=f"s{slab}-{patch}",original_image_id=f"o{slab}",specimen_id=f"s{slab}",slab_id=str(slab),image_path="x",mask_path="m",sha256=f"{slab:064x}",width_mm=150,height_mm=150))
        df = assign_slab_splits(pd.DataFrame(rows))
        self.assertEqual(df.groupby("split").size().to_dict(),dict(train=3584,val=768,test=768))
        validate_manifest(df,"segmentation")
        bad = df.copy(); bad.loc[0,"split"] = "test" if bad.loc[0,"split"] != "test" else "train"
        with self.assertRaisesRegex(ValueError,"leakage"): validate_manifest(bad,"segmentation")
        with self.assertRaisesRegex(ValueError,"overlap"): assert_independent(df,df.copy())
        other = df.copy()
        for col in ["image_id","original_image_id","specimen_id"]: other[col] = "b"+other[col]
        other.sha256 = "f"*64
        assert_independent(df,other)

    def test_crops_and_ids(self):
        windows = physical_windows(2400,2400,300,300)
        self.assertEqual(len(windows),16)
        self.assertEqual(windows[0],(0,0,1200,1200))
        self.assertEqual(windows[-1],(1200,1200,2400,2400))
        labels=np.array([[0,55],[900,7]],dtype=np.int32)
        self.assertEqual(set(np.unique(resize_labels(labels,(7,9)))),set(labels.ravel()))
        with self.assertRaises(ValueError):physical_windows(11,11,300,300)

    def test_feature_schema_and_empty(self):
        self.assertEqual(len(HEADERS),47);self.assertEqual(len(set(HEADERS)),47)
        result,detail,qa=measure(np.zeros((10,10),dtype=np.uint16),150,150,undefined="nan")
        self.assertEqual(result["N"],0);self.assertTrue(np.isnan(result["Dmin"]))
        self.assertFalse(detail);self.assertGreater(qa["undefined_count"],0)
        zero,_,_=measure(np.zeros((10,10),dtype=int),150,150,undefined="zero")
        self.assertTrue(all(np.isfinite(list(zero.values()))))
        default,_,qa=measure(np.zeros((10,10),dtype=int),150,150)
        self.assertEqual(default,zero)
        self.assertEqual(qa["undefined_policy"],"zero")
        with self.assertRaises(ValueError):measure(np.zeros((3,3)),150,150)

    def test_area_diameter_and_coarse_boundary(self):
        # One pixel is exactly a 2.36-mm equivalent disk by calibration.
        width=2.36*np.sqrt(np.pi)
        mask=np.array([[0,1],[0,0]],dtype=np.uint16)
        result,detail,_=measure(mask,width,width,undefined="nan")
        self.assertAlmostEqual(detail[0]["diameter_mm"],2.36)
        self.assertEqual(result["Nc"],1);self.assertEqual(result["N2.36"],1)
        self.assertAlmostEqual(result["A"],np.pi*2.36**2/4)
        self.assertEqual(result["Nedge"],1)
        self.assertTrue(np.isnan(result["Davg"]))
        interior,_,_=measure(mask,width,width,particles="interior")
        self.assertEqual(interior["N"],0);self.assertEqual(interior["Nedge"],1)

    def test_distance_and_disconnected_identity(self):
        mask=np.zeros((10,10),dtype=int);mask[2,2]=7;mask[7,7]=8;mask[2,3]=7
        out,detail,_=measure(mask,10,10)
        expected=np.hypot(7-2.5,7-2)
        self.assertAlmostEqual(out["Dmin"],expected)
        mask[9,9]=7
        out,detail,_=measure(mask,10,10)
        self.assertEqual(out["N"],2)
        self.assertEqual(detail[0]["components_8"],2)

    def test_independent_mask_set_and_overlap_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);one=np.zeros((8,8),dtype=np.uint8);two=one.copy()
            one[1:3,1:3]=255;two[5:7,5:7]=255
            Image.fromarray(one).save(root/"one.png");Image.fromarray(two).save(root/"two.png")
            descriptor=root/"instances.json"
            descriptor.write_text(json.dumps(dict(shape=[8,8],masks=["one.png","two.png"])))
            labels,paths=read_labels(descriptor)
            self.assertEqual(set(np.unique(labels)),{0,1,2});self.assertEqual(len(paths),3)
            descriptor.write_text(json.dumps(dict(shape=[8,8],masks=["one.png","one.png"])))
            with self.assertRaisesRegex(ValueError,"Overlapping"):read_labels(descriptor)

    def test_feature_failure_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);Image.fromarray(np.zeros((4,4,3),dtype=np.uint8)).save(root/"rgb.png")
            frame=pd.DataFrame([dict(image_id="bad",mask_path="rgb.png",width_mm=150,height_mm=150)])
            with self.assertRaises(ValueError):build_features(frame,root,root/"out")
            self.assertTrue((root/"out/errors.csv").exists())
            self.assertTrue((root/"out/.incomplete").exists())
            self.assertFalse((root/"out/manifest.json").exists())

    def test_strict_database_and_pairing(self):
        features=pd.DataFrame([{**dict.fromkeys(HEADERS,0),"image_id":"a"}])
        metadata=pd.DataFrame([dict(image_id="a",specimen_id="s",gradation_id="g",label_source="design",reference_id="lab-record")])
        labels=pd.DataFrame([dict(image_id="a",**dict(zip(TARGETS,[1,2,3,4,5,6])))])
        table=build_database(features,metadata,labels)
        self.assertEqual(len(table),1);self.assertEqual(table.N.iloc[0],0)
        align_tables([table,table.copy()])
        bad=table.copy();bad["P16"]=9
        with self.assertRaises(ValueError):align_tables([table,bad])
        with self.assertRaises(ValueError):build_database(features,metadata,labels.assign(image_id="wrong"))

if __name__ == "__main__":unittest.main()
