import gzip as gz
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from karyohmm import MetaHMM, PhaseCorrect, QuadHMM
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


def euploid_per_chrom(aneuploidy_df, names, chrom="chr1", pp_thresh=0.95):
    """Return only the euploid embryo names for this chromosome."""
    assert "bf_max_cat" in aneuploidy_df.columns
    assert "mother" in aneuploidy_df.columns
    assert "father" in aneuploidy_df.columns
    assert "child" in aneuploidy_df.columns
    assert "2" in aneuploidy_df.columns
    assert len(names) > 1
    filt_names = aneuploidy_df[
        (aneuploidy_df.child.isin(names))
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


if __name__ == "__main__":
    # Read in the input data and params ...
    aneuploidy_df = pd.read_csv(snakemake.input["aneuploidy_calls"], sep="\t")
    hmm = QuadHMM()
    hmm_dis = MetaHMM(disomy=True)
    family_data = load_baf_data(snakemake.input["baf_pkl"])
    names = [k for k in family_data.keys()]
    recomb_dict = {}
    lines = []
    for c in tqdm(snakemake.params["chroms"]):
        cur_names = euploid_per_chrom(aneuploidy_df, names, chrom=c, pp_thresh=snakemake.params["ppThresh"])
        mat_haps, pat_haps, bafs, real_names, pos = prep_data(
            family_dict=family_data, chrom=c, names=cur_names
        )
        nsibs = len(real_names)
        if nsibs >= 3:
            pi0_ests = np.zeros(nsibs)
            sigma_ests = np.zeros(nsibs)
            for i in range(nsibs):
                pi0_est, sigma_est = hmm_dis.est_sigma_pi0(
                    bafs=bafs[i][::5],
                    mat_haps=mat_haps[:, ::5],
                    pat_haps=pat_haps[:, ::5],
                    algo="Powell",
                    r=1e-4,
                )
                pi0_ests[i] = pi0_est
                sigma_ests[i] = sigma_est
            # Apply the phase correction method here ...
            phase_correct = PhaseCorrect(mat_haps=mat_haps, pat_haps=pat_haps)
            phase_correct.add_baf(bafs)
            phase_correct.phase_correct(
                pi0=np.median(pi0_ests), std_dev=np.median(sigma_ests)
            )
            phase_correct.phase_correct(
                maternal=False, pi0=np.median(pi0_ests), std_dev=np.median(sigma_ests)
            )
            recomb_dict[c] = {}
            # Use the least noisy siblings to help with estimation here ... 
            idxs = np.argsort(sigma_ests)
            for i in range(nsibs):
                paths0 = []
                for j in idxs:
                    if j != i:
                        path_ij = hmm.map_path(
                            bafs=[bafs[i], bafs[j]],
                            mat_haps=phase_correct.mat_haps_fixed,
                            pat_haps=phase_correct.pat_haps_fixed,
                            pi0=(pi0_ests[i], pi0_ests[j]),
                            std_dev=(sigma_ests[i], sigma_ests[j]),
                            r=1e-18,
                        )
                        paths0.append(path_ij)
                        # This ensures that the largest families still have reasonable runtimes
                        if len(paths0) > 2:
                            break
                # Isolate the recombinations here ...
                mat_rec, pat_rec, _, _ = hmm.isolate_recomb(paths0[0], paths0[1:])
                recomb_dict[c][f"{real_names[i]}"] = {
                    "pos": pos,
                    "paths": paths0,
                    "pi0_ests": pi0_ests,
                    "sigma_ests": sigma_ests,
                }
                for m in mat_rec:
                    _, left_pos, _, right_pos = find_nearest_het(
                        m, pos, phase_correct.mat_haps_fixed
                    )
                    rec_pos = pos[m]
                    lines.append(
                        f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tmaternal\t{left_pos}\t{rec_pos}\t{right_pos}\t{pi0_ests[i]}\t{sigma_ests[i]}\n'
                    )
                for p in pat_rec:
                    _, left_pos, _, right_pos = find_nearest_het(
                        p, pos, phase_correct.pat_haps_fixed
                    )
                    rec_pos = pos[p]
                    lines.append(
                        f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tpaternal\t{left_pos}\t{rec_pos}\t{right_pos}\t{pi0_ests[i]}\t{sigma_ests[i]}\n'
                    )
        else:
            pass
    # Write out the path dictionary with the viterbi traces
    pickle.dump(recomb_dict, gz.open(snakemake.output["recomb_paths"], "wb"))
    # Write out the formal crossover spot output here
    with open(snakemake.output["est_recomb"], "w") as out:
        out.write(
            "mother\tfather\tchild\tchrom\tcrossover_sex\tmin_pos\tavg_pos\tmax_pos\tavg_pi0\tavg_sigma\n"
        )
        for line in lines:
            out.write(line)
