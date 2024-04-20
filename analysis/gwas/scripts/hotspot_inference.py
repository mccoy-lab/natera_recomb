"""
Code implementing uncertainty quantification in hotspot-occupancy per-individual.
"""

import numpy as np 
from scipy.stats import norm
from tqdm import tqdm


def permute_rec_dist(r, nreps=100, std=200e3, seed=42):
    """Function to randomly shuffle a crossover."""
    pass

def est_co_overlap(rs, co_df):
    """Estimate the fraction of crossovers overlapping by chance."""
    pass

def est_p_overlap(rs, co_df, seed=42, **kwargs):
    assert seed > 0
    np.random.seed(seed)
    p_overlap = np.zeros(rs.shape[0])
    for i,r in enumerate(rs):
        rs_r = permute_rec_dist(r, **kwargs)
        p_overlap[i] = est_p_overlap(rs, co_df)
    return p_overlap

def loglik_co_overlap(alpha, p_overlaps, rs, co_df):
    loglik = 0.0
    pass

def est_mle_alpha(p_overlaps, rs, co_df, ngridpts=1000):
    """Output the 95% confidence interval of the MLE estimate for the proportion of crossovers at hotspots."""
    alphas = np.linspace(1e-4, 1-1e-4, ngridpts)
    pass
