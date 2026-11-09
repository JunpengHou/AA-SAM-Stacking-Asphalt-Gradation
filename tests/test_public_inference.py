import tempfile
from pathlib import Path
import unittest
import subprocess
import sys
import pandas as pd
import yaml
import numpy as np
from PIL import Image
from gradation.infer_sam3 import export_predictions

class PublicInferenceTests(unittest.TestCase):
    def test_export_identity_empty_and_mask_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/"synthetic.png";Image.new("RGB",(8,8)).save(source)
            def predictor(image):
                masks=np.zeros((2,8,8),bool);masks[0,1:4,1:4]=1;masks[1]=masks[0]
                return masks,np.array([.9,.8])
            rows=export_predictions([source],root/"output",predictor,"f"*64)
            self.assertEqual(rows[0]["n_instances"],1)
            with Image.open(root/"output/masks/00000.tif") as mask:self.assertEqual(np.asarray(mask).max(),1)
            with np.load(root/"output/candidates/00000.npz",allow_pickle=False) as data:
                self.assertEqual(data["checkpoint_sha256"].item(),"f"*64)
                self.assertEqual(len(data["scores"]),2)
            self.assertFalse((root/"output/.incomplete").exists())
            rows=export_predictions([source],root/"empty",lambda im:(np.zeros((0,8,8),bool),np.zeros(0)),"f"*64)
            self.assertEqual(rows[0]["n_instances"],0)
            with self.assertRaises(ValueError):export_predictions([source],root/"wrong",lambda im:(np.zeros((1,3,3),bool),np.ones(1)),"f"*64)

    def test_paired_model_comparison_and_mismatched_gt_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            truth=np.zeros((8,8),dtype=np.uint16);truth[1:4,1:4]=1
            Image.fromarray(truth).save(root/"truth.tif")
            np.savez(root/"candidates.npz",masks=np.array([truth>0]),scores=np.array([.9]))
            pd.DataFrame([dict(image_id="synthetic",truth_path="truth.tif",candidates_path="candidates.npz")]).to_csv(root/"samples.csv",index=False)
            for name in ["first","second"]:
                (root/(name+".yaml")).write_text(yaml.safe_dump(dict(samples="samples.csv",split="val",output="unused")))
            config=root/"comparison.yaml"
            config.write_text(yaml.safe_dump(dict(output="comparison",models={"SAM":"first.yaml","AA-SAM":"second.yaml"})))
            script=Path(__file__).resolve().parents[1]/"scripts/compare_segmentation.py"
            result=subprocess.run([sys.executable,str(script),"--config",str(config)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=pd.read_csv(root/"comparison/comparison.csv")
            self.assertEqual(len(report),2);self.assertTrue(report.Mask_AP50_all_points.eq(1.).all())
            bad=pd.read_csv(root/"samples.csv");bad.image_id="different";bad.to_csv(root/"bad.csv",index=False)
            (root/"second.yaml").write_text(yaml.safe_dump(dict(samples="bad.csv",split="val",output="unused")))
            config.write_text(yaml.safe_dump(dict(output="bad_comparison",models={"SAM":"first.yaml","AA-SAM":"second.yaml"})))
            result=subprocess.run([sys.executable,str(script),"--config",str(config)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("not exactly paired",result.stderr)
            self.assertFalse((root/"bad_comparison").exists())

if __name__=="__main__":unittest.main()
