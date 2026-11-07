import numpy as np


def best_threshold(y, p, cost=1.0, value=10.0, save_rate=0.3):
    '''Pick the cutoff that maximizes net value.
    cost: price of contacting one player. value: worth of a retained player.
    save_rate: share of contacted churners that outreach actually saves.'''
    best_t, best_net = 0.5, -1e18
    for t in np.linspace(0.05, 0.95, 91):
        flagged = p >= t
        saved = (flagged & (y == 1)).sum() * save_rate * value
        net = saved - flagged.sum() * cost
        if net > best_net:
            best_t, best_net = float(t), float(net)
    return best_t, best_net
