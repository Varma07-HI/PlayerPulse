import numpy as np
import shap


def make_explainer(model):
    return shap.TreeExplainer(model)


def top_reasons(explainer, X, features, k=3):
    sv = explainer.shap_values(X)
    out = []
    for impacts, row in zip(sv, X.values):
        idx = np.argsort(-np.abs(impacts))[:k]
        out.append([{
            'feature': features[i],
            'value': None if np.isnan(row[i]) else float(row[i]),
            'impact': float(impacts[i]),
        } for i in idx])
    return out