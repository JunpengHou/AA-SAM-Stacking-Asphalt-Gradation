"""Plot only computed run artifacts; no embedded manuscript metrics."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_figures(run,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run,output=Path(run),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    written=[]
    def save(fig,name):
        fig.tight_layout()
        for extension in ["svg","pdf","png"]:
            path=output/(name+"."+extension)
            fig.savefig(path,dpi=150)
            written.append(path)
        plt.close(fig)
    path=run/"outer_predictions.csv"
    if path.exists():
        pairs=pd.read_csv(path)
        for sieve,part in pairs.groupby("sieve"):
            if "S-LR" not in part:continue
            fig,ax=plt.subplots(figsize=(5,4))
            ax.scatter(part.reference,part["S-LR"])
            lo=min(part.reference.min(),part["S-LR"].min());hi=max(part.reference.max(),part["S-LR"].max())
            ax.plot([lo,hi],[lo,hi],color="grey",linestyle="--")
            ax.set(xlabel="Reference passing (%)",ylabel="Predicted passing (%)",
                   title=f"{sieve}: outer LOGO, {len(part)} design pairs")
            save(fig,"outer_"+sieve)
    for folder in sorted((run/"final").glob("P*")) if (run/"final").is_dir() else []:
        shap_path=folder/"final_stack_SHAP.csv";feature_path=folder/"final_stack_feature_values.csv"
        if not shap_path.exists() or not feature_path.exists():continue
        shap=pd.read_csv(shap_path).set_index("image_id")
        values=pd.read_csv(feature_path).set_index("image_id")
        if not shap.index.equals(values.index) or list(shap)!=list(values):raise ValueError("SHAP/feature rows or columns are not paired")
        order=shap.abs().mean().sort_values().tail(15).index
        importance=shap.abs().mean().sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(6,5));ax.barh(order,shap[order].abs().mean())
        ax.set(xlabel="Mean absolute permutation SHAP (percentage points)",title=f"{folder.name}: final fit, descriptive")
        save(fig,folder.name+"_shap_bar")
        fig,ax=plt.subplots(figsize=(7,5));rng=np.random.default_rng(2026)
        for i,name in enumerate(order):
            x=values[name].to_numpy(float)
            scale=np.ptp(x);color=(x-x.min())/scale if scale else np.zeros_like(x)
            ax.scatter(shap[name],i+rng.uniform(-.2,.2,len(shap)),c=color,cmap="coolwarm",vmin=0,vmax=1,s=9)
        ax.set_yticks(range(len(order)),order)
        ax.set(xlabel="Permutation SHAP (percentage points)",title=f"{folder.name}: final fit; color = relative feature value")
        save(fig,folder.name+"_shap_summary")
        for name in importance.head(3).index:
            fig,ax=plt.subplots(figsize=(5,4));ax.scatter(values[name],shap[name],s=12)
            ax.set(xlabel=name,ylabel="Permutation SHAP (percentage points)",title=f"{folder.name}: descriptive dependence")
            save(fig,folder.name+"_dependence_"+name)
    logs=[]
    for path in sorted(run.glob("outer_*/P*/feature_selection.csv.gz")):
        frame=pd.read_csv(path)
        frame=frame.loc[frame.fold.eq("refit")].copy()
        frame["sieve"]=path.parent.name;frame["outer_fold"]=path.parent.parent.name
        frame["removed"]=frame.decision.eq("constant_in_training")|frame.decision.str.startswith("redundant_with:")
        logs.append(frame)
    if logs:
        all_logs=pd.concat(logs)
        summary=all_logs.groupby(["sieve","feature"]).removed.agg(["sum","count","mean"]).reset_index()
        summary.rename(columns={"sum":"n_removed","count":"n_outer_fits","mean":"removal_frequency"}).to_csv(output/"feature_removal_frequency.csv",index=False)
    return written
