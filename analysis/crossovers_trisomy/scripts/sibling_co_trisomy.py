import gzip as gz
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import ruptures as rpt
from karyohmm import MetaHMM
from tqdm import tqdm


def bph(states):
    """Identify states that are BPH - both parental homologs."""
    idx = []
    for i, s in enumerate(states):
        assert len(s) == 4
        k = 0
        for j in range(4):
            k += s[j] >= 0
        if k == 3:
            if s[1] != -1:
                if s[0] != s[1]:
                    # Both maternal homologs present
                    idx.append(i)
            if s[3] != -1:
                if s[2] != s[3]:
                    # Both paternal homologs present
                    idx.append(i)
    # Returns indices of both maternal & paternal BPH
    return idx


def sph(states):
    """Identify states that are SPH - single parental homolog."""
    idx = []
    for i, s in enumerate(states):
        assert len(s) == 4
        k = 0
        for j in range(4):
            k += s[j] >= 0
        if k == 3:
            if s[1] != -1:
                if s[0] == s[1]:
                    # Both maternal homologs present
                    idx.append(i)
            if s[3] != -1:
                if s[2] == s[3]:
                    # Both paternal homologs present
                    idx.append(i)
    # Returns indices of both maternal & paternal SPH
    return idx


if __name__ == "__main__":
    # Read in the input data and params ...
    chrom = snakemake.wildcards["chrom"]
    trisomy_df = pl.read_csv(
        snakemake.input["trisomy_calls"], null_values=["NA"], separator="\t"
    )
    baf_data = pickle.load(gz.open(snakemake.input["baf_pkl"], "rb"))
    hmm_data = pickle.load(gz.open(snakemake.input["hmm_pkl"], "rb"))
    mother = snakemake.wildcards["mother"]
    father = snakemake.wildcards["father"]
    child = snakemake.wildcards["child"]
    hmm = MetaHMM()
    filt_df = (
        trisomy_df.filter(pl.col("mother") == mother)
        .filter(pl.col("father") == father)
        .filter(pl.col("child") == child)
        .filter(pl.col("chrom") == chrom)
    )
    call = filt_df["bf_max_cat"].to_numpy().astype(str)[0]
    pi0_est = filt_df["pi0_baf"].to_numpy()[0]
    sigma_est = filt_df["sigma_baf"].to_numpy()[0]
    if call == "3m":
        hmm.states = hmm.m_trisomy_states
        hmm.karyotypes = np.array(["3m", "3m", "3m", "3m", "3m", "3m"], dtype=str)
    elif call == "3p":
        hmm.states = hmm.p_trisomy_states
        hmm.karyotypes = np.array(["3p", "3p", "3p", "3p", "3p", "3p"], dtype=str)
    else:
        raise ValueError("Chromosome is not determined to be a trisomy!")
    gammas, states, karyotypes = hmm.forward_backward(
        bafs=baf_data[chrom]["baf_embryo"],
        pos=baf_data[chrom]["pos"],
        mat_haps=baf_data[chrom]["mat_haps"],
        pat_haps=baf_data[chrom]["pat_haps"],
        pi0=pi0_est,
        std_dev=sigma_est,
        unphased=False,
    )
    gammas_bph_sph = np.vstack(
        [
            np.exp(gammas)[bph(states), :].sum(axis=0),
            np.exp(gammas)[sph(states), :].sum(axis=0),
        ]
    )
    # Use changepoint detection on the posterior trace to estimate BPH -> SPH shifts
    pts = rpt.KernelCPD(min_size=snakemake.params["min_size"]).fit_predict(
        gammas_bph_sph.T, pen=snakemake.params["penalty"]
    )
    # Write out the formal crossover spot output here
    lines = []
    for p in pts[:-1]:
        co_sex = "maternal" if call == "3m" else "paternal"
        if (p > 0) and (p < gammas_bph_sph.shape[1]):
            min_pos = baf_data[chrom]["pos"][p - 1]
            avg_pos = baf_data[chrom]["pos"][p]
            max_pos = baf_data[chrom]["pos"][p + 1]
            lines.append(
                f"{mother}\t{father}\t{child}\t{chrom}\t{co_sex}\t{min_pos}\t{avg_pos}\t{max_pos}\t{pi0_est}\t{sigma_est}\n"
            )
    with open(snakemake.output["trisomy_recomb"], "w") as out:
        out.write(
            "mother\tfather\tchild\tchrom\tcrossover_sex\tmin_pos\tavg_pos\tmax_pos\tavg_pi0\tavg_sigma\n"
        )
        if len(lines) == 0:
            out.write(
                f"{mother}\t{father}\t{child}\t{chrom}\tNA\tNA\tNA\tNA\t{pi0_est}\t{sigma_est}\n"
            )
        else:
            for line in lines:
                out.write(line)
