"""
Code implementing uncertainty quantification in hotspot-occupancy per-individual.

Based on proposed model from Coop et al 2008.
"""

import gzip as gz

import numpy as np
import pandas as pd
import polars as pl
from intervaltree import IntervalTree
from scipy.stats import norm
from tqdm import tqdm


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
    p_overlap = np.zeros(len(rs))
    for i, r in enumerate(rs):
        rs_r = permute_rec_dist(r, **kwargs)
        p_overlap[i] = est_co_overlap(rs_r, co_int_dict)
    return p_overlap


def est_deltas(rs, co_int_dict):
    """Get an indicator of whether the hotspot overlaps a known hotspot.

    NOTE: here the value for `rs` is the actual realized set of crossovers.

    """
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
        loglik += np.log(d * x + (1 - d) * (1 - x))
    return loglik


def est_mle_alpha(p_overlaps, deltas, ngridpts=500):
    """Output the 95% confidence interval of the MLE estimate for the proportion of crossovers at hotspots."""
    assert ngridpts >= 50
    assert p_overlaps.size == deltas.size
    alphas = np.linspace(1e-6, 1 - 1e-6, ngridpts)
    llgrid = np.array([loglik_co_overlap(a, p_overlaps, deltas) for a in alphas])
    mle_alpha = alphas[np.argmax(llgrid)]
    lower_95 = np.min(alphas[llgrid >= (np.max(llgrid) - 2.0)])
    upper_95 = np.max(alphas[llgrid >= (np.max(llgrid) - 2.0)])
    assert lower_95 <= mle_alpha
    assert mle_alpha <= upper_95
    return (lower_95, mle_alpha, upper_95)


if __name__ == "__main__":
    """Actually do the full estimation routine across crossovers for each meiosis."""

    co_df = pd.read_csv(snakemake.input["crossover_fp"], sep="\t")
    male_hotspot_df = pd.read_csv(snakemake.input["male_hotspots"], sep="\t")
    female_hotspot_df = pd.read_csv(snakemake.input["female_hotspots"], sep="\t")
    # pratto_hotspot_df = pd.read_csv(snakemake.input["pratto_hotspots"], sep="\t")

    # Step 0: setup the parameters
    max_interval = snakemake.params["max_interval"]
    nreps = snakemake.params["nreps"]
    ngridpts = snakemake.params["ngridpts"]

    # Step 1: Make a dictionary of the sex-specific hotspots from Halldorsson 2019
    co_hotspot_dict_male = create_co_intervals(male_hotspot_df)
    co_hotspot_dict_female = create_co_intervals(female_hotspot_df)

    tuple_mat_df = (
        co_df[co_df.crossover_sex == "maternal"]
        .groupby("uid")[["chrom", "min_pos", "max_pos"]]
        .apply(lambda x: list(tuple(x.values)))
        .reset_index()
    )
    tuple_mat_df.rename(columns={0: "co_list"}, inplace=True)

    tuple_pat_df = (
        co_df[co_df.crossover_sex == "paternal"]
        .groupby("uid")[["chrom", "min_pos", "max_pos"]]
        .apply(lambda x: list(tuple(x.values)))
        .reset_index()
    )
    tuple_pat_df.rename(columns={0: "co_list"}, inplace=True)

    # Step 2: Inference for maternal crossovers ...
    res_mat = []
    for u, x in tqdm(zip(tuple_mat_df.uid, tuple_mat_df.co_list)):
        co_mat_calls = [(c, s, e) for (c, s, e) in x if (e - s) < max_interval]
        p_overlaps_prime = est_p_overlap(
            co_mat_calls, co_hotspot_dict_female, nreps=nreps
        )
        deltas_prime = est_deltas(co_mat_calls, co_hotspot_dict_female)
        alphas = est_mle_alpha(p_overlaps_prime, deltas_prime, ngridpts=ngridpts)
        if len(co_mat_calls) > 0:
            res_mat.append(
                [
                    u,
                    alphas[0],
                    alphas[1],
                    alphas[2],
                    len(co_mat_calls),
                    np.unique([x[0] for x in co_mat_calls]).size,
                ]
            )
        else:
            res_mat.append(
                [
                    u,
                    np.nan,
                    np.nan,
                    np.nan,
                    len(co_mat_calls),
                    0,
                ]
            )
    res_mat_df = pd.DataFrame(
        res_mat,
        columns=[
            "uid",
            "lower_95_alpha_mat",
            "mean_alpha_mat",
            "upper_95_alpha_mat",
            "nco_pass_mat",
            "nchrom",
        ],
    )

    # Step 3: Inference for paternal crossovers ...
    res_pat = []
    for u, x in tqdm(zip(tuple_pat_df.uid, tuple_pat_df.co_list)):
        co_pat_calls = [(c, s, e) for (c, s, e) in x if (e - s) < max_interval]
        p_overlaps_prime = est_p_overlap(
            co_pat_calls, co_hotspot_dict_male, nreps=nreps
        )
        deltas_prime = est_deltas(co_pat_calls, co_hotspot_dict_male)
        alphas = est_mle_alpha(p_overlaps_prime, deltas_prime, ngridpts=ngridpts)
        if len(co_pat_calls) > 0:
            res_pat.append(
                [
                    u,
                    alphas[0],
                    alphas[1],
                    alphas[2],
                    len(co_pat_calls),
                    np.unique([x[0] for x in co_pat_calls]).size,
                ]
            )
        else:
            res_pat.append(
                [
                    u,
                    np.nan,
                    np.nan,
                    np.nan,
                    len(co_pat_calls),
                    0,
                ]
            )
    res_pat_df = pd.DataFrame(
        res_pat,
        columns=[
            "uid",
            "lower_95_alpha_pat",
            "mean_alpha_pat",
            "upper_95_alpha_pat",
            "nco_pass_pat",
            "nchrom",
        ],
    )
    # Merge in data on covariates here using polars
    co_df = pl.from_pandas(co_df)
    print(co_df.columns)
    aneuploidy_df = pl.read_csv(
        snakemake.input["aneuploidy_tsv"], separator="\t", null_values=["NA"]
    )
    aneuploidy_df = aneuploidy_df.with_columns(
        pl.concat_str(
            [
                pl.col("mother"),
                pl.col("father"),
                pl.col("child"),
            ],
            separator="+",
        ).alias("uid"),
    )
    genmap_df = pl.read_csv(
        snakemake.input["genmap"], comment_prefix="#", separator="\t"
    )
    genlen_df = (
        genmap_df.group_by(pl.col("Chr"))
        .agg(pl.col("cM").max(), pl.col("End").max(), pl.col("Begin").min())
        .with_columns((pl.col("End") - pl.col("Begin")).alias("seq_len"))
        .rename({"Chr": "chrom", "cM": "cM_len"})
        .sort("seq_len")
    )
    covar_df = pl.read_csv(
        snakemake.input["covariates"], null_values="NA", separator="\t"
    )
    res_mat_df = pl.from_pandas(res_mat_df)
    res_pat_df = pl.from_pandas(res_pat_df)
    res_mat_filt_df = (
        res_mat_df.join(
            co_df[
                [
                    "uid",
                    "mother",
                    "father",
                    "child",
                    "euploid",
                    "maternal_meiotic_aneuploidy",
                ]
            ].unique(),
            on=["uid"],
        )
        .with_columns(pl.col("mother").alias("IID"))
        .join(covar_df, on=["IID"])
    )
    res_pat_filt_df = (
        res_pat_df.join(
            co_df[
                [
                    "uid",
                    "mother",
                    "father",
                    "child",
                    "euploid",
                    "maternal_meiotic_aneuploidy",
                ]
            ].unique(),
            on=["uid"],
        )
        .with_columns(pl.col("father").alias("IID"))
        .join(covar_df, on=["IID"])
    )
    if snakemake.params["euploid"]:
        res_pat_filt_df = res_pat_filt_df.filter(pl.col("euploid"))
        res_mat_filt_df = res_mat_filt_df.filter(pl.col("euploid"))
    else:
        res_pat_filt_df = res_pat_filt_df.filter(~pl.col("euploid"))
        res_mat_filt_df = res_mat_filt_df.filter(~pl.col("euploid"))
    with gz.open(snakemake.output["maternal_occupancy"], "w+") as mat_out:
        res_mat_filt_df.write_csv(mat_out, null_value="NA")
    with gz.open(snakemake.output["paternal_occupancy"], "w+") as pat_out:
        res_pat_filt_df.write_csv(pat_out, null_value="NA")
