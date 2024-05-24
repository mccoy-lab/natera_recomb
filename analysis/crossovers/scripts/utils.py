import gzip as gz
import pickle
import sys
from pathlib import Path
from karyohmm import MetaHMM
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_baf_data(baf_pkls):
    """Load in the multiple BAF datasets."""
    family_data = {}
    for fp in baf_pkls:
        embryo_name = Path(fp).stem.split(".")[0]
        with gz.open(fp, "rb") as f:
            data = pickle.load(f)
            family_data[embryo_name] = data
    return family_data


def euploid_per_chrom(
    aneuploidy_df, mother, father, names, chrom="chr1", pp_thresh=0.95
):
    """Return only the euploid embryo names for this chromosome."""
    assert "bf_max_cat" in aneuploidy_df.columns
    assert "mother" in aneuploidy_df.columns
    assert "father" in aneuploidy_df.columns
    assert "child" in aneuploidy_df.columns
    assert "2" in aneuploidy_df.columns
    assert len(names) > 1
    filt_names = aneuploidy_df[
        (aneuploidy_df.mother == mother)
        & (aneuploidy_df.father == father)
        & (aneuploidy_df.child.isin(names))
        & (aneuploidy_df.chrom == chrom)
        & (aneuploidy_df["2"] >= pp_thresh)
    ].child.values
    if filt_names.size < 3:
        return []
    else:
        return filt_names.tolist()


def prep_data(family_dict, names, chrom="chr21"):
    """Prepare the data for the chromosome to have the same length in BAF across all samples."""
    shared_pos = []
    for k in family_dict.keys():
        if k in names:
            shared_pos.append(family_dict[k][chrom]["pos"])
    bafs = []
    real_names = []
    mat_haps = None
    pat_haps = None
    pos = None
    try:
        collective_pos = list(set(shared_pos[0]).intersection(*shared_pos))
        for k in family_dict.keys():
            if k in names:
                cur_pos = family_dict[k][chrom]["pos"]
                baf = family_dict[k][chrom]["baf_embryo"]
                idx = np.isin(cur_pos, collective_pos)
                bafs.append(baf[idx])
                mat_haps = family_dict[k][chrom]["mat_haps"][:, idx]
                pat_haps = family_dict[k][chrom]["pat_haps"][:, idx]
                real_names.append(k)
        pos = np.sort(collective_pos)
    except IndexError:
        # NOTE: this only happens when we have all embryos as aneuploid for the chromosome.
        pass
    return mat_haps, pat_haps, bafs, real_names, pos


def est_genotype_quality(hmm_data, pos, chrom="chr21"):
    """Estimate the genotype quality by the local estimate of disomy at each SNP across the sibling embryos."""
    assert len(hmm_data) > 0
    assert pos.size > 0
    hmm = MetaHMM()
    posterior_disomy_agg = []
    for k in hmm_data:
        cur_pos = hmm_data[k][chrom]["pos"]
        g, karyo = hmm.marginal_posterior_karyotypes(
            hmm_data[k][chrom]["gammas"], hmm_data[k][chrom]["karyotypes"]
        )
        assert g.shape[1] == cur_pos.size
        cur_posterior_disomy = g[np.where(karyo == "2")[0], np.isin(cur_pos, pos)]
        posterior_disomy_agg.append(cur_posterior_disomy)
    posterior_disomy = np.mean(np.vstack(posterior_disomy_agg), axis=0)
    assert posterior_disomy.size == pos.size
    return posterior_disomy


def extract_parameters(aneuploidy_df, mother, father, names, chrom):
    """Extract the core parameters for inference from the aneuploidy data frame."""
    filt_df = aneuploidy_df[
        (aneuploidy_df.mother == mother)
        & (aneuploidy_df.father == father)
        & (aneuploidy_df.chrom == chrom)
    ]
    pi0_bafs = np.zeros(len(names))
    sigma_bafs = np.zeros(len(names))
    for i, n in enumerate(names):
        pi0_baf_test = filt_df[(filt_df.child == n)].pi0_baf.values[0]
        sigma_baf_test = filt_df[(filt_df.child == n)].sigma_baf.values[0]
        pi0_bafs[i] = pi0_baf_test
        sigma_bafs[i] = sigma_baf_test
    assert np.all(pi0_bafs > 0)
    assert np.all(sigma_bafs > 0)
    return pi0_bafs, sigma_bafs


def find_nearest_het(idx, pos, haps):
    """Find the nearest heterozygotes to the estimated crossover position."""
    assert idx >= 0 and idx <= haps.shape[1]
    assert pos.size == haps.shape[1]
    geno_focal = haps[0, :] + haps[1, :]
    het_idx = np.where((geno_focal == 1))[0]
    if idx < np.min(het_idx):
        left_pos = np.nan
        left_idx = np.nan
    else:
        try:
            left_idx = het_idx[het_idx < idx][-1]
            left_pos = pos[left_idx]
        except IndexError:
            # For recombinations at the very beginning of chromosomes
            left_idx = 0
            left_pos = pos[0]
    if idx > np.max(het_idx):
        right_idx = np.nan
        right_pos = np.nan
    else:
        try:
            right_idx = het_idx[het_idx >= idx][0]
            right_pos = pos[right_idx]
        except IndexError:
            right_idx = pos.size - 1
            right_pos = pos[-1]
    return left_idx, left_pos, right_idx, right_pos
