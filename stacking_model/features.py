"""Fold-local preprocessing and permutation SHAP (interventional background)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def permutation_shap(predict, x, background, repeats, seed):
    """Monte Carlo Shapley values via forward and reverse feature orders.
    One independent training-background draw per query and permutation pair.
    Vectorized prediction across the complete path for each order.
    """
    rng = np.random.default_rng(seed)
    n, p = x.shape
    contributions = np.zeros((n, p))
    for _ in range(repeats):
        ref = background[rng.integers(len(background), size=n)]
        order = rng.permutation(p)
        for perm in (order, order[::-1]):
            path = np.tile(ref, (p+1, 1, 1))
            for step, col in enumerate(perm, 1):
                path[step:, :, col] = x[:, col]
            values = predict(path.reshape(-1, p)).reshape(p+1, n)
            contributions[:, perm] += np.diff(values, axis=0).T
    return contributions / (2*repeats)

def rank_features(x, y, names, args):
    # An entirely missing training column receives 0 and is then marked constant.
    medians = pd.DataFrame(x).median().fillna(0).to_numpy()
    z = np.where(np.isnan(x), medians, x)
    constant = np.ptp(z, axis=0) == 0
    if constant.all():
        raise ValueError("All features are constant in a training fold.")
    pilot = RandomForestRegressor(n_estimators=args.pilot_trees, min_samples_leaf=2,
                                  random_state=args.seed, n_jobs=1).fit(z, y)
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(len(z), min(args.shap_samples, len(z)), replace=False)
    phi = permutation_shap(pilot.predict, z[sample], z, args.permutations, args.seed)
    importance = np.abs(phi).mean(axis=0)
    df = pd.DataFrame(z, columns=names)
    corr = df.corr()
    # Correlation with a constant target is undefined; retain NaN in the report.
    with np.errstate(invalid="ignore", divide="ignore"):
        target_corr = df.corrwith(pd.Series(y)).to_numpy()
    report = pd.DataFrame({"feature": names, "mean_abs_SHAP": importance,
                           "Pearson_with_target": target_corr,
                           "constant": constant})
    return medians, corr, report

def choose(corr, report, threshold):
    if threshold is None:
        return list(range(len(report))), ["all_features"]*len(report)
    scores = report.mean_abs_SHAP.to_numpy()
    keep, reason = [], [""]*len(report)
    for j in np.argsort(-scores, kind="stable"):
        if report.constant.iloc[j]:
            reason[j] = "constant_in_training"
            continue
        match = next((k for k in keep if abs(corr.iat[j,k]) >= threshold), None)
        if match is None:
            keep.append(int(j))
            reason[j] = "retained"
        else:
            reason[j] = "redundant_with:" + str(report.feature.iloc[match])
    return sorted(keep), reason

def transform(x, medians, columns):
    return np.where(np.isnan(x), medians, x)[:,columns]
