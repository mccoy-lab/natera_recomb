import gzip as gz
import pickle
import sys
from pathlib import Path

import numpy as np
from karyohmm import QuadHMM
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


def prepare_paired_data(
    embryo_id1="10013440016_R06C01",
    embryo_id2="10013440016_R04C01",
    embryo_id3="10013440016_R05C01",
    chrom="chr21",
    data_dict=None,
):
    """Create the filtered dataset for evaluating crossovers using the QuadHMM results."""
    data_embryo1 = data_dict[embryo_id1][chrom]
    data_embryo2 = data_dict[embryo_id2][chrom]
    data_embryo3 = data_dict[embryo_id3][chrom]
    if (data_embryo1["pos"].size != data_embryo2["pos"].size) or (
        (data_embryo2["pos"].size != data_embryo3["pos"].size)
    ):
        pos1 = data_embryo1["pos"]
        pos2 = data_embryo2["pos"]
        pos3 = data_embryo3["pos"]
        idx2 = np.isin(pos2, pos1) & np.isin(pos2, pos3)
        idx1 = np.isin(pos1, pos2) & np.isin(pos1, pos3)
        idx3 = np.isin(pos3, pos1) & np.isin(pos3, pos2)
        baf1 = data_embryo1["baf_embryo"][idx1]
        baf2 = data_embryo2["baf_embryo"][idx2]
        baf3 = data_embryo3["baf_embryo"][idx3]
        mat_haps = data_embryo1["mat_haps"][:, idx1]
        pat_haps = data_embryo1["pat_haps"][:, idx1]
        assert baf1.size == baf2.size
        assert baf2.size == baf3.size
        # Return the maternal haplotypes, paternal haplotypes, baf
        return mat_haps, pat_haps, baf1, baf2, baf3, pos1[idx1]
    else:
        pos = data_embryo1["pos"]
        baf1 = data_embryo1["baf_embryo"]
        baf2 = data_embryo2["baf_embryo"]
        baf3 = data_embryo3["baf_embryo"]
        mat_haps = data_embryo1["mat_haps"]
        pat_haps = data_embryo1["pat_haps"]
        # Return the maternal haplotypes, paternal haplotypes, baf
        assert baf1.size == baf2.size
        assert baf2.size == baf3.size
        return mat_haps, pat_haps, baf1, baf2, baf3, pos


def find_nearest_het(idx, pos, haps_focal, haps_alt):
    """Find the nearest heterozygotes to the estimated crossover position."""
    assert idx >= 0 and idx <= haps_focal.shape[1]
    assert pos.size == haps_focal.shape[1]
    geno_focal = haps_focal[0, :] + haps_focal[1, :]
    geno_alt = haps_alt[0, :] + haps_alt[1, :]
    het_idx = np.where((geno_focal == 1) & ((geno_alt == 0) | (geno_alt == 2)))[0]
    if idx < np.min(het_idx):
        left_pos = np.nan
        left_idx = np.nan
    else:
        left_idx = het_idx[het_idx <= idx][-1]
        left_pos = pos[left_idx]
    if idx > np.max(het_idx):
        right_idx = np.nan
        right_pos = np.nan
    else:
        right_idx = het_idx[het_idx >= idx][0]
        right_pos = pos[right_idx]
    return left_idx, left_pos, right_idx, right_pos


if __name__ == "__main__":
    # Read in the input data and params ...
    hmm = QuadHMM()
    family_data = load_baf_data(snakemake.input["baf_pkl"])
    names = [k for k in family_data.keys()]
    nsibs = len(names)
    recomb_dict = {}
    lines = []
    for c in tqdm(snakemake.params["chroms"]):
        recomb_dict[c] = {}
        for i in range(nsibs):
            j = (i + 1) % nsibs
            j2 = (i + 2) % nsibs
            mat_haps, pat_haps, baf0, baf1, baf2, pos = prepare_paired_data(
                embryo_id1=names[i],
                embryo_id2=names[j],
                embryo_id3=names[j2],
                chrom=c,
                data_dict=family_data,
            )
            pi0_01, sigma_01 = hmm.est_sigma_pi0(
               bafs=[baf0, baf1], mat_haps=mat_haps, pat_haps=pat_haps, r=1e-18
            )
            path_01, _, _, _ = hmm.viterbi_algorithm(
                bafs=[baf0, baf1],
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_01,
                std_dev=sigma_01,
                r=1e-18,
            )
            refined_path_01 = hmm.restrict_path(path_01)
            pi0_02, sigma_02 = hmm.est_sigma_pi0(
               bafs=[baf0, baf2], mat_haps=mat_haps, pat_haps=pat_haps, r=1e-18
            )
            path_02, _, _, _ = hmm.viterbi_algorithm(
                bafs=[baf0, baf2],
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_02,
                std_dev=sigma_02,
                r=1e-18,
            )
            refined_path_02 = hmm.restrict_path(path_02)
            mat_rec, pat_rec = hmm.isolate_recomb(
                refined_path_01, refined_path_02, window=20
            )
            recomb_dict[c][f'{names[i]}+{names[j]}+{names[j2]}'] = {'pos': pos, 'path_01': refined_path_01, 'path_02': refined_path_02, 'pi0_01': pi0_01, 'pi0_02': pi0_02, 'sigma_01': sigma_01, 'sigma_02': sigma_02}
            for m in mat_rec:
                _, left_pos, _, right_pos = find_nearest_het(m[0], pos, mat_haps, pat_haps)
                rec_pos = pos[m[0]]
                lines.append(
                    f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{names[i]}\t{c}\tmaternal\t{left_pos}\t{rec_pos}\t{right_pos}\t{np.mean([pi0_01, pi0_02])}\t{np.mean([sigma_01, sigma_02])}\n'
                )
            for p in pat_rec:
                _, left_pos, _,  right_pos = find_nearest_het(p[0], pos, pat_haps, mat_haps)
                rec_pos = pos[p[0]]
                lines.append(
                    f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{names[i]}\t{c}\tpaternal\t{left_pos}\t{rec_pos}\t{right_pos}\t{np.mean([pi0_01, pi0_02])}\t{np.mean([sigma_01, sigma_02])}\n'
                )
    # Write out the path dictionary with the viterbi traces 
    pickle.dump(recomb_dict, gz.open(snakemake.output["recomb_paths"], "wb"))
    # Write out the formal crossover spot output here 
    with open(snakemake.output["est_recomb"], "w") as out:
        out.write(
            "mother\tfather\tchild\tchrom\tcrossover_sex\tmin_pos\tavg_pos\tmax_pos\tavg_pi0\tavg_sigma\n"
        )
        for line in lines:
            out.write(line)
