import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
from stacking_model.train import FEATURES,TARGETS,splits,fit_one,load_table,metrics


def synthetic_table(groups=10,images=3):
    rng=np.random.default_rng(7)
    rows=[]
    for group in range(groups):
        for image in range(images):
            row=dict(zip(FEATURES,rng.normal(size=47)+group/4))
            row.update(image_id=f"g{group:02}-i{image}",specimen_id=f"g{group:02}-s{image}",gradation_id=f"g{group:02}")
            row.update(dict(zip(TARGETS,np.arange(1.,7.)*10+group)))
            rows.append(row)
    return pd.DataFrame(rows)


class StackingTests(unittest.TestCase):
    def test_18_outer_groups_and_inner(self):
        groups=np.repeat(np.arange(18),80)
        seen=[]
        for group in np.unique(groups):
            dev=np.flatnonzero(groups!=group);test=np.flatnonzero(groups==group)
            self.assertEqual(len(test),80);seen.extend(test.tolist())
            for tr,va in splits(dev,groups,5):
                self.assertFalse(set(groups[tr])&set(groups[va]));self.assertNotIn(group,groups[tr]);self.assertNotIn(group,groups[va])
        self.assertEqual(sorted(seen),list(range(1440)))

    def test_meta_provenance_and_outer_label_invariance(self):
        df=synthetic_table();g=df.gradation_id.to_numpy()
        test=np.flatnonzero(g=="g09");dev=np.flatnonzero(g!="g09")
        args=Namespace(folds=2,rules=[None],models=["DT","SVR"],max_base=2,trials=4,seed=12,pilot_trees=2,shap_samples=2,permutations=1)
        with tempfile.TemporaryDirectory() as temp:
            first=Path(temp)/"first";second=Path(temp)/"second";first.mkdir();second.mkdir()
            fit_one(df,"P13.2",dev,test,first,args)
            changed=df.copy();changed.loc[test,"P13.2"]=9999
            changed.loc[test,FEATURES]=100000
            fit_one(changed,"P13.2",dev,test,second,args)
            self.assertEqual((first/"config.json").read_bytes(),(second/"config.json").read_bytes())
            self.assertEqual((first/"meta_oof.csv").read_bytes(),(second/"meta_oof.csv").read_bytes())
            self.assertEqual(len(pd.read_csv(first/"meta_oof.csv")),len(dev))
            provenance=json.loads((first/"oof_provenance.json").read_text())
            expected_ids=set(df.image_id.iloc[dev])
            development={}
            for row in provenance:
                self.assertFalse(set(row["fit_groups"])&set(row["predicted_groups"]))
                self.assertNotIn("g09",row["fit_groups"]+row["predicted_groups"])
                self.assertTrue(set(row["fit_image_ids"])<=expected_ids)
                if row["stage"]=="inner-meta":
                    self.assertFalse(set(row["excluded_groups"])&set(row["fit_groups"]+row["predicted_groups"]))
                else:development.setdefault(row["model"],[]).extend(row["predicted_image_ids"])
            for ids in development.values():self.assertEqual(sorted(ids),sorted(expected_ids))

    def test_group_mean_metrics_and_near_zero(self):
        # Two equally weighted group means, independent of image counts.
        out=metrics(np.array([10.,20.]),np.array([12.,18.]))
        self.assertEqual(out["MSE"],4.)
        self.assertAlmostEqual(out["R2"],.84)
        df=synthetic_table();df.loc[0,"P0.075"]=1e-12
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"table.csv";df.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"near-zero"):load_table(path)

if __name__=="__main__":unittest.main()
