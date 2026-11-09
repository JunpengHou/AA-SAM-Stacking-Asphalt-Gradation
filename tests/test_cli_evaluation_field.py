import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from gradation.database import TARGETS

class EvaluationFieldCliTests(unittest.TestCase):
    def test_evaluation_and_field_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            truth=np.zeros((12,12),dtype=np.uint16);truth[2:7,2:7]=1
            Image.fromarray(truth).save(root/"truth.tif")
            np.savez(root/"candidates.npz",masks=np.array([truth>0]),scores=np.array([.9]))
            pd.DataFrame([dict(image_id="synthetic",truth_path="truth.tif",candidates_path="candidates.npz")]).to_csv(root/"samples.csv",index=False)
            configuration=dict(samples="samples.csv",split="val",synthetic=True,output="evaluation")
            (root/"eval.yaml").write_text(yaml.safe_dump(configuration))
            result=subprocess.run([sys.executable,"-m","gradation.cli","eval-seg","--config",str(root/"eval.yaml")],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            output=json.loads((root/"evaluation/segmentation_metrics.json").read_text())
            self.assertEqual(output["Mask_AP50_all_points"],1.)
            self.assertEqual(output["Boundary_F1_GT_weighted"],1.)
            patches=[];cores=[]
            for point in range(4):
                ids=dict(road_id="synthetic-road",cross_section_id="section",point_id=f"p{point}")
                cores.append({**ids,**dict(zip(TARGETS,[10.,20.,30.,40.,50.,60.]))})
                for patch in range(4):patches.append({**ids,"patch_id":f"x{patch}",**dict(zip(TARGETS,[11.,21.,31.,41.,51.,61.]))})
            pd.DataFrame(patches).to_csv(root/"patches.csv",index=False);pd.DataFrame(cores).to_csv(root/"cores.csv",index=False)
            (root/"field.yaml").write_text(yaml.safe_dump(dict(patch_predictions="patches.csv",cores="cores.csv",output="field",synthetic=True)))
            result=subprocess.run([sys.executable,"-m","gradation.cli","validate-field","--config",str(root/"field.yaml")],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            metrics=pd.read_csv(root/"field/cross_section_metrics.csv")
            self.assertEqual(len(metrics),1);self.assertAlmostEqual(metrics.MAE.iloc[0],1.)
            self.assertAlmostEqual(metrics.MAPE.iloc[0],np.mean(1/np.arange(10,61,10))*100)
            provenance=json.loads((root/"field/command_manifest.json").read_text())
            self.assertIn(str(root/"cores.csv"),provenance["inputs"])

if __name__=="__main__":unittest.main()
