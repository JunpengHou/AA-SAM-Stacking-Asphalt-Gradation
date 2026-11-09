import json
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
import pandas as pd
import yaml
from gradation.smoke import run_smoke
from gradation.feature47 import HEADERS

class CliIntegrationTests(unittest.TestCase):
    def test_feature_schema_and_cpu_pipeline(self):
        registry=yaml.safe_load((Path(__file__).resolve().parents[1]/"feature_schema.yaml").read_text(encoding="utf-8"))
        self.assertEqual([f["name"] for f in registry["features"]],HEADERS)
        self.assertEqual(pd.Series([f["category"] for f in registry["features"]]).value_counts().to_dict(),dict(area=29,count=16,spatial=2))
        with tempfile.TemporaryDirectory() as folder:
            root=run_smoke(Path(folder)/"synthetic",targets=["P13.2"])
            features=pd.read_csv(root/"features/features.csv")
            self.assertEqual(list(features.columns),["image_id",*HEADERS])
            self.assertEqual(len(features),30)
            outer=pd.read_csv(root/"stacking/outer_predictions.csv")
            self.assertEqual(len(outer),10)
            self.assertEqual(outer.gradation_id.nunique(),10)
            prediction=pd.read_excel(root/"predictions.xlsx")
            self.assertEqual(list(prediction.columns),["image_id","P13.2"])
            self.assertEqual(len(prediction),30)
            manifest=json.loads((root/"smoke_results.json").read_text())
            self.assertFalse(manifest["research_replication"])
            self.assertTrue((root/"report/figures/outer_P13.2.svg").exists())
            self.assertTrue((root/"report/figures/P13.2_shap_summary.pdf").exists())
            self.assertEqual(len(pd.read_csv(root/"comparison/fixture_a_eta_categories.csv")),3)
            self.assertTrue(all(r["returncode"]==0 for r in manifest["runs"]))
            # Mutation of version identity is rejected, not silently used.
            wrong=pd.read_csv(root/"database/database.csv");wrong["feature_schema_version"]="wrong"
            wrong.to_csv(root/"wrong.csv",index=False)
            result=subprocess.run([sys.executable,"-m","stacking_model.predict",str(root/"wrong.csv"),"--run",str(root/"stacking"),"--out",str(root/"wrong_predictions.xlsx"),"--targets","P13.2"],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("Feature schema mismatch",result.stderr)
            self.assertFalse((root/"wrong_predictions.xlsx").exists())

if __name__=="__main__":unittest.main()
