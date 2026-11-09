"""Plot descriptive full-dataset Pearson correlations and final model SHAP values."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
if __package__:
    from .train import FEATURES, TARGETS, load_table
else:
    from train import FEATURES, TARGETS, load_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--run", required=True, type=Path)
    opts = ap.parse_args()
    df = load_table(opts.input)
    folder = opts.run/"figures"
    folder.mkdir(exist_ok=True, parents=True)
    names = FEATURES + TARGETS
    corr = df[names].corr()
    corr.to_csv(folder/"descriptive_Pearson.csv")
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)), names, fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(folder/"descriptive_Pearson.pdf")
    plt.close(fig)
    for target in TARGETS:
        f = opts.run/"final"/target/"final_stack_importance.csv"
        if not f.exists():
            continue
        scores = pd.read_csv(f,index_col=0).iloc[:,0].head(15).sort_values()
        fig, ax = plt.subplots(figsize=(6,5))
        ax.barh(scores.index, scores, color="#376D89")
        ax.set_xlabel("Mean absolute SHAP value (percentage points)")
        ax.set_title(f"{target[1:]} mm sieve")
        fig.tight_layout()
        fig.savefig(folder/f"SHAP_{target}.pdf")
        plt.close(fig)
if __name__=="__main__":
    main()
