"""Sieve-specific nested grouped stacking. See README_zh.md for the data contract."""
import argparse
from itertools import combinations
from pathlib import Path
import hashlib
import json
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

if __package__:
    from .learners import make_model, parameter_grid, SPACES
    from .features import rank_features, choose, transform, permutation_shap
else:
    from learners import make_model, parameter_grid, SPACES
    from features import rank_features, choose, transform, permutation_shap

TARGETS = ["P0.075", "P2.36", "P4.75", "P9.5", "P13.2", "P16"]
FEATURES = """N0 N2.36 N4.75 N9.5 N13.2 N16 Nc Nf N Rn0 Rn2.36 Rn4.75 Rn9.5 Rn13.2 Rn16 Nedge A A0 A2.36 A4.75 A9.5 A13.2 A16 A0_avg A2.36_avg A4.75_avg A9.5_avg A13.2_avg A16_avg Aavg Ra0 Ra2.36 Ra4.75 Ra9.5 Ra13.2 Ra16 Ac Ac_avg Af Af_avg Racf Ra Ras Aedge Redge Davg Dmin""".split()
IDS = ["image_id", "specimen_id", "gradation_id"]
def load_table(path, mape_min=1e-8):
    df = pd.read_excel(path, dtype=object) if path.suffix.lower()==".xlsx" else pd.read_csv(path, dtype=object)
    missing = set(IDS+FEATURES+TARGETS)-set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for col in IDS:
        df[col] = df[col].astype("string").str.strip()
        if df[col].isna().any() or df[col].eq("").any():
            raise ValueError(f"Missing {col}")
    if df.image_id.duplicated().any() or df.groupby("specimen_id").gradation_id.nunique().max()!=1:
        raise ValueError("Image IDs must be unique and specimen IDs globally identify one design.")
    df[FEATURES+TARGETS] = df[FEATURES+TARGETS].apply(pd.to_numeric, errors="raise").astype(float)
    y = df[TARGETS].to_numpy()
    if np.isinf(df[FEATURES]).any().any() or not np.isfinite(y).all() or (y<=mape_min).any() or (y>100).any():
        raise ValueError("Features may be missing but not infinite. MAPE requires targets above the declared near-zero threshold and at most 100%.")
    if (np.diff(y,axis=1)<0).any() or (df.groupby("gradation_id")[TARGETS].nunique()>1).any().any():
        raise ValueError("Reference gradations must be nondecreasing and constant within design.")
    return df.sort_values(IDS).reset_index(drop=True)

def splits(indices, groups, n):
    if n < 2:raise ValueError("Grouped OOF requires at least two folds")
    if len(np.unique(groups[indices])) < n:
        raise ValueError(f"Insufficient training gradations for {n} folds.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate split indices")
    for tr, va in GroupKFold(n).split(indices, groups=groups[indices]):
        fit, hold = indices[tr], indices[va]
        assert not set(groups[fit]) & set(groups[hold])
        yield fit, hold

def group_errors(y, pred, groups):
    v = pd.DataFrame({"y": y, "p": pred, "g": groups}).groupby("g")[["y","p"]].mean()
    return ((v.p-v.y).abs()/v.y*100).to_numpy()

def bayes_search(name, folds, y, g, args, min_train):
    # GP surrogate + expected improvement on a finite, ordered hyperparameter grid.
    from sklearn.exceptions import ConvergenceWarning
    grid, coords = parameter_grid(name, min_train)
    rng = np.random.default_rng(args.seed)
    seen, losses, history = [], [], []
    best = (np.inf, None, None)
    for trial in range(min(args.trials,len(grid))):
        remaining = np.setdiff1d(np.arange(len(grid)), seen)
        if trial < min(3, args.trials):
            j = int(rng.choice(remaining))
        else:
            gp = GaussianProcessRegressor(kernel=Matern(length_scale=0.5,nu=2.5),
                alpha=1e-6,normalize_y=True,random_state=args.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",ConvergenceWarning)
                gp.fit(coords[seen], losses)
            mu, sd = gp.predict(coords[remaining], return_std=True)
            sd = np.maximum(sd, 1e-12)
            improvement = min(losses)-mu-0.001
            ei = improvement*norm.cdf(improvement/sd)+sd*norm.pdf(improvement/sd)
            j = int(remaining[np.argmax(ei)])
        oof = np.full(len(y),np.nan)
        for tr, va, xtr, xva in folds:
            model = make_model(name,grid[j],args.seed).fit(xtr,y[tr])
            oof[va] = model.predict(xva)
        use = np.isfinite(oof)
        expected = np.concatenate([f[1] for f in folds])
        if not use[expected].all():
            raise ValueError(f"{name}: non-finite validation prediction")
        loss = float(np.mean(group_errors(y[use],oof[use],g[use])))
        seen.append(j); losses.append(loss)
        history.append((trial+1,json.dumps(grid[j]),loss))
        if loss < best[0]:
            best = loss, grid[j], oof
    return *best, history

def metrics(y,p):
    return dict(MSE=mean_squared_error(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),
                MAE=mean_absolute_error(y,p),R2=r2_score(y,p) if np.ptp(y)>0 else np.nan,
                MAPE=np.mean(np.abs((y-p)/y))*100)

def fit_one(df, target, dev, test, folder, args, finalists=None):
    x = df[FEATURES].to_numpy(float); y = df[target].to_numpy(float)
    g = df.gradation_id.to_numpy(str)
    folds = list(splits(dev,g,args.folds))
    # To score a stack on an inner validation fold, build LR training inputs only
    # from that fold's training gradations, via a second round of grouped OOF fits.
    subfolds = [list(splits(tr,g,min(args.folds,len(np.unique(g[tr]))))) for tr,_ in folds]
    smallest = min(len(t) for sub in subfolds for t,_ in sub)
    cache, audit, corr_logs, trials, rankings = {}, [], [], [], []
    oof_provenance, selected_oof = [], None
    def ranking(tr, label):
        key = tuple(tr)
        if key not in cache:
            med,corr,report = rank_features(x[tr],y[tr],FEATURES,args)
            cache[key] = med,corr,report
            if not label.startswith("sub"):
                block = corr.copy();block.insert(0,"feature",block.index);block.insert(0,"fold",label)
                corr_logs.append(block)
        return cache[key]
    def prep(tr,va,rule,label,log=True):
        med,corr,report = ranking(tr,label)
        cols,why = choose(corr,report,rule)
        if log:
            audit.append(report.assign(fold=label,rule=str(rule),decision=why))
        return tr,va,transform(x[tr],med,cols),transform(x[va],med,cols)
    best_stack, best_singles = None, {}
    for rule in args.rules:
        fold_data = [prep(tr,va,rule,f"inner-{i+1}") for i,(tr,va) in enumerate(folds)]
        sub_data = [[prep(tr,va,rule,f"sub-{i}-{j}",False) for j,(tr,va) in enumerate(sub)]
                    for i,sub in enumerate(subfolds)]
        oof = np.full((len(y),len(args.models)),np.nan)
        sub_oof = [np.full_like(oof,np.nan) for _ in folds]
        params = {}
        for j,name in enumerate(args.models):
            score,pars,pred,hist = bayes_search(name,fold_data,y,g,args,smallest)
            params[name]=pars
            trials.extend(("development",str(rule),name,*row) for row in hist)
            if name not in best_singles or score < best_singles[name][0]:
                best_singles[name] = score,rule,pars
            for i,sub in enumerate(sub_data):
                inner_tr, inner_va = folds[i]
                local_min = min(len(t) for t, *_ in sub)
                _, local_pars, _, local_hist = bayes_search(name,sub,y,g,args,local_min)
                trials.extend((f"inner-{i+1}",str(rule),name,*row) for row in local_hist)
                # Held-out inner group labels never enter the parameter search.
                _, _, xtr, xva = fold_data[i]
                oof[inner_va,j] = make_model(name,local_pars,args.seed).fit(xtr,y[inner_tr]).predict(xva)
                for k,(tr,va,xtr,xva) in enumerate(sub):
                    tune = list(splits(tr,g,min(args.folds,len(np.unique(g[tr])))))
                    tune_data = [prep(t,v,rule,f"sub-tune-{i}-{k}",False) for t,v in tune]
                    _, sub_pars, _, sub_hist = bayes_search(name,tune_data,y,g,args,min(len(t) for t,_ in tune))
                    trials.extend((f"inner-{i+1}-sub-{k+1}",str(rule),name,*row) for row in sub_hist)
                    sub_oof[i][va,j] = make_model(name,sub_pars,args.seed).fit(xtr,y[tr]).predict(xva)
                    oof_provenance.append(dict(stage="inner-meta", rule=rule, model=name,
                        fit_image_ids=df.image_id.iloc[tr].tolist(), predicted_image_ids=df.image_id.iloc[va].tolist(),
                        fit_groups=sorted(set(g[tr])), predicted_groups=sorted(set(g[va])),
                        excluded_groups=sorted(set(g[inner_va])), parameters=sub_pars))
                oof_provenance.append(dict(stage="development-meta",rule=rule,model=name,
                    fit_image_ids=df.image_id.iloc[inner_tr].tolist(),predicted_image_ids=df.image_id.iloc[inner_va].tolist(),
                    fit_groups=sorted(set(g[inner_tr])),predicted_groups=sorted(set(g[inner_va])),parameters=local_pars))
        for size in range(2,min(len(args.models),args.max_base)+1):
            for combo in combinations(range(len(args.models)),size):
                if finalists is not None and tuple(args.models[j] for j in combo) not in finalists:
                    continue
                pred = np.full(len(y),np.nan)
                for i,(tr,va) in enumerate(folds):
                    lr = LinearRegression().fit(sub_oof[i][tr][:,combo],y[tr])
                    pred[va] = lr.predict(oof[va][:,combo])
                errs = group_errors(y[dev],pred[dev],g[dev])
                residuals = pd.DataFrame(oof[dev][:,combo]-y[dev,None]).groupby(g[dev]).mean()
                corr = residuals.corr().abs().to_numpy()
                upper = corr[np.triu_indices(size,1)]
                redundancy = np.nan_to_num(upper,nan=1.0).mean()
                key = (float(errs.mean()),float(errs.std()),float(redundancy),size)
                names = [args.models[j] for j in combo]
                rankings.append((str(rule),",".join(names),*key))
                if best_stack is None or key < best_stack[0]:
                    meta = LinearRegression().fit(oof[dev][:,combo],y[dev])
                    best_stack = key,rule,names,{n:params[n] for n in names},meta
                    selected_oof = pd.DataFrame(oof[dev][:,combo], columns=names)
    key,rule,names,pars,meta = best_stack
    median,corr,report = ranking(dev,"refit")
    cols,why = choose(corr,report,rule)
    audit.append(report.assign(fold="refit",rule=str(rule),decision=why))
    z = transform(x[dev],median,cols)
    fitted = {name:make_model(name,pars[name],args.seed).fit(z,y[dev]) for name in names}
    def stack_predict(v):
        return meta.predict(np.column_stack([m.predict(v) for m in fitted.values()]))
    selected_oof.insert(0,"image_id",df.image_id.iloc[dev].to_numpy())
    selected_oof.insert(1,"gradation_id",g[dev])
    selected_oof.to_csv(folder/"meta_oof.csv",index=False)
    (folder/"oof_provenance.json").write_text(json.dumps(oof_provenance,indent=2),encoding="utf-8")
    config = {"protocol_version":"strict-nested-oof-v2",
              "feature_schema_version":getattr(args,"feature_schema_version","unversioned-legacy"),
              "feature_policy":getattr(args,"feature_policy",{}), "rule":rule,"features":[FEATURES[i] for i in cols],"base":pars,
              "LR_coefficients":meta.coef_.tolist(),"LR_intercept":float(meta.intercept_),
              "inner_MAPE":key[0],"individual":best_singles}
    (folder/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    pd.concat(audit).to_csv(folder/"feature_selection.csv.gz",index=False)
    pd.concat(corr_logs).to_csv(folder/"pearson.csv.gz",index=False)
    pd.DataFrame(trials,columns=["scope","rule","model","trial","parameters","MAPE"]).to_csv(folder/"bayes.csv",index=False)
    pd.DataFrame(rankings,columns=["rule","combination","MAPE","APE_SD","abs_residual_corr","n_models"]).to_csv(folder/"stack_selection.csv",index=False)
    split_log=[{"inner":i+1,"training":sorted(set(g[tr])),"validation":sorted(set(g[va]))}
               for i,(tr,va) in enumerate(folds)]
    (folder/"splits.json").write_text(json.dumps(split_log,indent=2,ensure_ascii=False),encoding="utf-8")
    if test is None:
        joblib.dump(dict(columns=cols,median=median,base=fitted,meta=meta,features=FEATURES,
                         feature_schema_version=getattr(args,"feature_schema_version","unversioned-legacy"),
                         feature_policy=getattr(args,"feature_policy",{})),folder/"model.joblib")
        take=np.random.default_rng(args.seed).choice(len(z),min(len(z),args.shap_samples),replace=False)
        phi=permutation_shap(stack_predict,z[take],z,args.permutations,args.seed)
        detail=pd.DataFrame(phi,columns=config["features"])
        detail.insert(0,"image_id",df.image_id.iloc[dev[take]].to_numpy())
        detail.to_csv(folder/"final_stack_SHAP.csv",index=False)
        feature_values=pd.DataFrame(z[take],columns=config["features"])
        feature_values.insert(0,"image_id",df.image_id.iloc[dev[take]].to_numpy())
        feature_values.to_csv(folder/"final_stack_feature_values.csv",index=False)
        pd.Series(np.abs(phi).mean(axis=0),index=config["features"],name="mean_abs_SHAP").sort_values(ascending=False).to_csv(folder/"final_stack_importance.csv")
    else:
        out=df.loc[test,IDS].copy()
        out["sieve"]=target; out["reference"]=y[test]
        out["S-LR"]=stack_predict(transform(x[test],median,cols))
        for name,(_,r,p) in best_singles.items():
            if name in fitted and rule==r and pars[name]==p:
                out[name]=fitted[name].predict(transform(x[test],median,cols))
            else:
                c,_=choose(corr,report,r)
                model=make_model(name,p,args.seed).fit(transform(x[dev],median,c),y[dev])
                out[name]=model.predict(transform(x[test],median,c))
        if not np.isfinite(out[["S-LR",*args.models]].to_numpy()).all():
            raise ValueError("Non-finite outer prediction.")
        out.to_csv(folder/"predictions.csv",index=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input",type=Path); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--models",default="RF,XGBoost,SVR,DT,KNN,MLP,LSTM")
    ap.add_argument("--targets",default=",".join(TARGETS))
    ap.add_argument("--rules",default="all,0.95",help="No pruning or absolute inter-feature correlation thresholds")
    ap.add_argument("--trials",type=int,default=10)
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--max-base",type=int,default=7)
    ap.add_argument("--shap-samples",type=int,default=32)
    ap.add_argument("--permutations",type=int,default=8)
    ap.add_argument("--pilot-trees",type=int,default=64)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--mape-min",type=float,default=1e-8,help="Reject zero/near-zero targets; no epsilon substitution")
    ap.add_argument("--final-models",help="Retain a development-selected combination for final fitting; default: modal outer development choices, ties via full-data inner CV")
    ap.add_argument("--skip-final",action="store_true",help="Run outer evaluation only")
    ap.add_argument("--paper-layout",action="store_true",help="Require 18 designs, 20 specimens/design and 4 images/specimen")
    a=ap.parse_args()
    a.models=a.models.split(",");targets=a.targets.split(",")
    if len(set(a.models))!=len(a.models) or len(a.models)<2 or set(a.models)-SPACES.keys():
        ap.error("Select at least two distinct supported learners: "+",".join(SPACES))
    if set(targets)-set(TARGETS) or len(set(targets))!=len(targets):
        ap.error("Invalid or duplicated sieve target")
    if a.final_models:
        fixed=a.final_models.split(",")
        if len(set(fixed))!=len(fixed) or len(fixed)<2 or set(fixed)-set(a.models):
            ap.error("final-models must be at least two distinct candidate learners")
    a.rules=[None if s=="all" else float(s) for s in a.rules.split(",")]
    if any(r is not None and not 0<r<=1 for r in a.rules):ap.error("Thresholds must be in (0,1].")
    if a.folds<2 or a.max_base<2 or min(a.trials,a.permutations,a.shap_samples,a.pilot_trees)<1:
        ap.error("Invalid search or sampling budget")
    if a.trials<4:ap.error("At least 4 trials are needed for the GP-guided phase.")
    if "XGBoost" in a.models:import xgboost
    if "LSTM" in a.models:import torch
    if not np.isfinite(a.mape_min) or a.mape_min < 0:ap.error("mape-min must be finite and nonnegative")
    df=load_table(a.input,a.mape_min);groups=df.gradation_id.to_numpy(str);designs=np.unique(groups)
    development_groups=len(designs)-1
    inner_training_groups=development_groups-int(np.ceil(development_groups/a.folds))
    sub_folds=min(a.folds,inner_training_groups)
    smallest_tuning_groups=(inner_training_groups-int(np.ceil(inner_training_groups/sub_folds))) if sub_folds>=2 else 0
    if len(designs)<=a.folds or smallest_tuning_groups<2:
        ap.error("Too few designs for strict nested grouped tuning/OOF; increase independent groups or revise the declared fold count.")
    if a.paper_layout and (len(designs)!=18 or not df.groupby("gradation_id").specimen_id.nunique().eq(20).all()
        or not df.groupby("specimen_id").size().eq(4).all()):
        ap.error("Expected 18 gradations × 20 specimens × 4 images")
    if "feature_schema_version" in df:
        versions_in_table=df.feature_schema_version.dropna().unique()
        if len(versions_in_table)!=1 or df.feature_schema_version.isna().any():
            ap.error("Exactly one feature schema version is required")
        a.feature_schema_version=str(versions_in_table[0])
    else:
        a.feature_schema_version="unversioned-legacy"
        warnings.warn("Unversioned features: legacy compatibility only; do not mix with the 2.36-mm schema")
    a.feature_policy={}
    for field in ["feature_undefined_policy","feature_particles_policy"]:
        if field in df:
            if df[field].isna().any() or df[field].nunique()!=1:ap.error(f"Mixed feature policy: {field}")
            a.feature_policy[field]=str(df[field].iloc[0])
    a.out.mkdir(parents=True,exist_ok=True)
    args_dict={k:v for k,v in vars(a).items() if k not in {"input","out","skip_final"}}
    import importlib.metadata as md
    versions={name:md.version(name) for name in ["numpy","pandas","scipy","scikit-learn","joblib"]+
              (["xgboost"] if "XGBoost" in a.models else [])+(["torch"] if "LSTM" in a.models else [])}
    sha=hashlib.sha256()
    for f in ["train.py","features.py","learners.py"]:
        sha.update(Path(__file__).with_name(f).read_bytes())
    stamp=json.dumps(dict(settings=args_dict,versions=versions,code_sha256=sha.hexdigest(),
                    input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),designs=designs.tolist()),
                    sort_keys=True,ensure_ascii=False,indent=2)
    manifest=a.out/"manifest.json"
    if manifest.exists() and manifest.read_text(encoding="utf-8")!=stamp:
        ap.error("Inputs, settings or code changed. Choose a new --out directory.")
    if not manifest.exists() and any(a.out.iterdir()):ap.error("Choose an empty output directory.")
    if not manifest.exists():manifest.write_text(stamp,encoding="utf-8")
    parts=[]
    for i in range(len(designs)+(not a.skip_final)):
        final=i==len(designs)
        fold="final" if final else f"outer_{i+1:02d}"
        test=None if final else np.flatnonzero(groups==designs[i])
        dev=np.arange(len(df)) if final else np.flatnonzero(groups!=designs[i])
        for target in targets:
            folder=a.out/fold/target
            if not folder.exists():
                tmp=folder.with_name(target+".partial")
                # Only one process should write to a particular output directory.
                if tmp.exists():shutil.rmtree(tmp)
                tmp.mkdir(parents=True)
                print(f"{fold} / {target}",flush=True)
                finalists = None
                if final:
                    from collections import Counter
                    if a.final_models:
                        retained=set(a.final_models.split(","))
                        finalists={tuple(name for name in a.models if name in retained)}
                    else:
                        # Manuscript p22: retain development-selected combination,
                        # retune parameters on all laboratory groups, never use outer errors.
                        counts=Counter()
                        for outer in range(len(designs)):
                            choice=json.loads((a.out/f"outer_{outer+1:02d}"/target/"config.json").read_text(encoding="utf-8"))
                            counts[tuple(choice["base"])]+=1
                        finalists={combo for combo,n in counts.items() if n==max(counts.values())}
                fit_one(df,target,dev,test,tmp,a,finalists)
                tmp.rename(folder)
            if not final:
                parts.append(pd.read_csv(folder/"predictions.csv",dtype={c:str for c in IDS}))
    images=pd.concat(parts,ignore_index=True)
    models=["S-LR",*a.models]
    grp=images.groupby(["sieve","gradation_id"],sort=False)[["reference",*models]].mean().reset_index()
    rows=[]
    for sieve,tab in grp.groupby("sieve",sort=False):
        for m in models:rows.append(dict(sieve=sieve,model=m,n_designs=len(tab),**metrics(tab.reference,tab[m])))
    grp.insert(0,"outer_fold",grp.gradation_id.map({d:i+1 for i,d in enumerate(designs)}))
    counts=images.groupby(["sieve","gradation_id"]).size().rename("n_images").reset_index()
    grp=grp.merge(counts,on=["sieve","gradation_id"],validate="one_to_one")
    residuals=grp.melt(id_vars=["outer_fold","sieve","gradation_id","reference","n_images"],value_vars=models,var_name="model",value_name="prediction")
    residuals["residual"]=residuals.prediction-residuals.reference
    residuals["APE_percent"]=abs(residuals.residual/residuals.reference)*100
    residuals.to_csv(a.out/"residuals.csv",index=False)
    images.to_csv(a.out/"image_predictions.csv",index=False)
    grp.to_csv(a.out/"outer_predictions.csv",index=False)
    result=pd.DataFrame(rows)
    result.to_csv(a.out/"metrics_by_sieve.csv",index=False)
    avg=result.groupby("model",sort=False)[["MSE","RMSE","MAE","R2","MAPE"]].agg(lambda values: values.mean(skipna=False)).reset_index()
    avg.to_csv(a.out/"mean_over_sieves.csv",index=False)
    tmp=a.out/"results.partial.xlsx"
    with pd.ExcelWriter(tmp,engine="openpyxl") as w:
        avg.to_excel(w,sheet_name="Mean_over_sieves",index=False)
        result.to_excel(w,sheet_name="Metrics_by_sieve",index=False)
        grp.to_excel(w,sheet_name="LOGO_gradations",index=False)
        images.to_excel(w,sheet_name="LOGO_images",index=False)
    tmp.replace(a.out/"results.xlsx")
    print(a.out/"results.xlsx")

if __name__=="__main__":
    main()
