"""Override for the e22 16:6 protocol: 6 evaluation planners per draw (RandomState(1000+draw))."""
import numpy as np
R_DRAWS = 16
def unified_split(seed, utypes, n_planners=16):
    rng = np.random.RandomState(1000 + seed)
    hp = sorted(rng.choice(n_planners, 4, replace=False).tolist())
    ht = set(np.array(sorted(utypes))[rng.choice(len(utypes), 8, replace=False)].tolist())
    return hp, ht
