import json

import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.config import FEATURES, MODEL_DIR, PROC_DIR
from src.splits import time_split


def main():
    torch.manual_seed(0)
    df = pd.read_parquet(PROC_DIR / 'features.parquet')
    tr, va, te = time_split(df)
    imp, sc = SimpleImputer(strategy='median'), StandardScaler()
    sc.fit(imp.fit_transform(tr[FEATURES]))

    def tensor(d):
        return torch.tensor(sc.transform(imp.transform(d[FEATURES])), dtype=torch.float32)

    Xtr, Xva, Xte = tensor(tr), tensor(va), tensor(te)
    ytr = torch.tensor(tr.churn.values, dtype=torch.float32)

    net = nn.Sequential(nn.Linear(len(FEATURES), 32), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    def predict(X):
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(X)).squeeze(1).numpy()

    best, best_state = -1, None
    for epoch in range(30):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]).squeeze(1), ytr[idx])
            loss.backward()
            opt.step()
        pr = average_precision_score(va.churn, predict(Xva))
        if pr > best:
            best, best_state = pr, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    mlp_test = average_precision_score(te.churn, predict(Xte))
    xgb_test = json.loads((MODEL_DIR / 'meta.json').read_text())['test_metrics']['pr_auc']
    print(f'Test PR-AUC  MLP: {mlp_test:.4f}   XGBoost: {xgb_test:.4f}')


if __name__ == '__main__':
    main()