import gzip as gz
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from karyohmm import MetaHMM
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


def euploid_per_chrom(aneuploidy_df, names, chrom="chr1"):
    """Return only the euploid embryo names for this chromosome."""
    assert "bf_max_cat" in aneuploidy_df.columns
    assert "mother" in aneuploidy_df.columns
    assert "father" in aneuploidy_df.columns
    assert "child" in aneuploidy_df.columns
    assert len(names) > 1
    filt_names = aneuploidy_df[
        (aneuploidy_df.child.isin(names))
        & (aneuploidy_df.chrom == chrom)
        & (aneuploidy_df.bf_max_cat == "2")
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
    collective_pos = list(set(shared_pos[0]).intersection(*shared_pos))
    bafs = []
    real_names = []
    for k in family_dict.keys():
        if k in names:
            cur_pos = family_dict[k][chrom]["pos"]
            baf = family_dict[k][chrom]["baf_embryo"]
            idx = np.isin(cur_pos, collective_pos)
            print(idx.size)
            bafs.append(baf[idx])
            mat_haps = family_dict[k][chrom]["mat_haps"][:, idx]
            pat_haps = family_dict[k][chrom]["pat_haps"][:, idx]
            real_names.append(k)
    pos = np.sort(collective_pos)
    return mat_haps, pat_haps, bafs, real_names, pos


if __name__ == "__main__":
    # Read in the input data and params ...
    aneuploidy_df = pd.read_csv(snakemake.input["aneuploidy_calls"], sep="\t")
    hmm_dis = MetaHMM(disomy=True)
    family_data = load_baf_data(snakemake.input["baf_pkl"])
    names = [k for k in family_data.keys()]
    recomb_dict = {}
    lines = []
    for c in tqdm(snakemake.params["chroms"]):
        cur_names = euploid_per_chrom(aneuploidy_df, names, chrom=c)
        nsibs = len(cur_names)
        print(c, nsibs, cur_names)
        if nsibs >= 3:
            recomb_dict[c] = {}
            mat_haps, pat_haps, bafs, real_names, pos = prep_data(
                family_dict=family_data, chrom=c, names=cur_names
            )
            print(len(bafs), cur_names, real_names)
            for i in range(len(real_names)):
                pi0_est, sigma_est = hmm_dis.est_sigma_pi0(
                    bafs=bafs[i][::5],
                    mat_haps=mat_haps[:, ::5],
                    pat_haps=pat_haps[:, ::5],
                    algo="Powell",
                    r=1e-4,
                )
                # Write out the lines appropriately
                lines.append(
                    f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\t{pi0_est}\t{sigma_est}\n'
                )
        else:
            pass
    # Write out the formal crossover spot output here
    with open(snakemake.output["est_params"], "w") as out:
        out.write("mother\tfather\tchild\tchrom\tpi0_hat\tsigma_hat\n")
        for line in lines:
            out.write(line)
