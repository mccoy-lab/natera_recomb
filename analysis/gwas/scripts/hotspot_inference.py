"""
Code implementing uncertainty quantification in hotspot-occupancy per-individual.

Based on proposed model from Coop et al 2008.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from intervaltree import IntervalTree


def create_co_intervals(co_df):
    """Store hotspot intervals in a dictionary of interval trees."""
    assert "chrom" in co_df.columns
    assert "start" in co_df.columns
    assert "end" in co_df.columns
    uniq_chroms = np.unique(co_df.chrom.values)
    hotspot_interval_dict = {}
    for c in uniq_chroms:
        cur_hotspots = [
            (s, e)
            for (s, e) in zip(
                co_df[co_df.chrom == c].start.values, co_df[co_df.chrom == c].end.values
            )
        ]
        cur_interval_tree = IntervalTree.from_tuples(cur_hotspots)
        hotspot_interval_dict[c] = cur_interval_tree
    return hotspot_interval_dict


def permute_rec_dist(r, nreps=100, std=200e3, seed=None):
    """Function to randomly shuffle a crossover.

    Arguments:
        - r : triplet specifying chrom, start, end of crossover location
        - nreps : number of replicates by which to shuffle the
        - std: standard deviation of shuffle in recombination position in bp
        - seed: random seed
    """
    assert nreps > 0
    assert std > 0
    if seed is not None:
        assert seed > 0
        np.random.seed(seed)
    chrom, start, end = r
    eps = norm.rvs(scale=std, size=nreps)
    r_prime = [(chrom, start + e, end + e) for e in eps]
    return r_prime


def est_co_overlap(rs, co_int_dict):
    """Estimate the fraction of crossovers overlapping by chance.

    NOTE: here the `rs` are the **resampled** intervals

    """
    n_tot = len(rs)
    n_overlap = 0
    for chrom, start, end in rs:
        x = co_int_dict[chrom].overlap(start, end)
        n_overlap += int(len(x) > 0)
    return n_overlap / n_tot


def est_p_overlap(rs, co_int_dict, seed=42, **kwargs):
    """Estimate the probability of overlap"""
    assert seed > 0
    np.random.seed(seed)
    p_overlap = np.zeros(rs.shape[0])
    for i, r in enumerate(rs):
        rs_r = permute_rec_dist(r, **kwargs)
        p_overlap[i] = est_co_overlap(rs_r, co_int_dict)
    return p_overlap


def est_deltas(rs, co_int_dict):
    """Get an indicator of whether the hotspot overlaps a known hotspot.

    NOTE: here the value for `rs` is the actual realized set of crossovers.

    """
    assert len(rs) > 0
    deltas = np.zeros(len(rs), dtype=int)
    for i, (chrom, start, end) in enumerate(rs):
        x = co_int_dict[chrom].overlap(start, end)
        deltas[i] = int(len(x) > 0)
    return deltas


def loglik_co_overlap(alpha, p_overlaps_chance, deltas):
    """Calculate the log-likelihood of the crossover data under a given alpha parameter for hotspot occupancy."""
    assert (alpha > 0) and (alpha < 1)
    assert p_overlaps_chance.size == deltas.size
    loglik = 0.0
    for p, d in zip(p_overlaps_chance, deltas):
        x = alpha + (1 - alpha) * p
        loglik += np.log(x + (1 - d) * (1 - x))
    return loglik


def est_mle_alpha(p_overlaps, deltas, ngridpts=500):
    """Output the 95% confidence interval of the MLE estimate for the proportion of crossovers at hotspots."""
    assert ngridpts >= 50
    assert p_overlaps.size == deltas.size
    alphas = np.linspace(1e-4, 1 - 1e-4, ngridpts)
    llgrid = np.array([loglik_co_overlap(a, p_overlaps, deltas) for a in alphas])
    mle_alpha = alphas[np.argmax(llgrid)]
    lower_95 = np.min(alphas[llgrid >= (np.max(llgrid) - 2.0)])
    upper_95 = np.max(alphas[llgrid >= (np.max(llgrid) - 2.0)])
    assert lower_95 <= mle_alpha
    assert mle_alpha <= upper_95
    return (lower_95, mle_alpha, upper_95)


if __name__ == "__main__":
    """Actually do the full estimation routine across the crossovers for some samples."""
    co_df = pd.read_csv(snakemake.input["co_data"], sep="\t")
    hotspot_df = pd.read_csv(snakemake.input["hotspots"], sep="\t")
    # Step 1: Make a dictionary of the various hotspots
    hotspot_chrom_dict = create_co_intervals(hotspot_df)
    co_df['uid'] = co_df['mother'] + co_df['father'] + co_df['child']
    for uid in np.unique(co_df.uid.values):
        cur_mat_df = co_df[(co_df.uid == uid) & (co_df.crossover_sex == "maternal")]
        cur_pat_df = co_df[(co_df.uid == uid) & (co_df.crossover_sex == "paternal")]
        # Obtain the maternal copies

    pass
