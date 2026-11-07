import numpy as np


def time_split(df, gap=3):
    c = np.sort(df['cutoff'].unique())
    n = len(c)
    a, b = int(n * 0.6), int(n * 0.8)
    train_c = c[:a]
    valid_c = c[a + gap:b]
    test_c = c[b + gap:]
    pick = lambda cs: df[df['cutoff'].isin(cs)]
    return pick(train_c), pick(valid_c), pick(test_c)
