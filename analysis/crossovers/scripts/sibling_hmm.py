import gzip as gz
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from karyohmm import MetaHMM, PhaseCorrect, QuadHMM
from tqdm import tqdm
from utils import *


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
        cur_names = euploid_per_chrom(
            aneuploidy_df,
            mother=snakemake.wildcards["mother"],
            father=snakemake.wildcards["father"],
            names=names,
            chrom=c,
            pp_thresh=snakemake.params["ppThresh"],
        )
        mat_haps, pat_haps, bafs, real_names, pos = prep_data(
            family_dict=family_data, chrom=c, names=cur_names
        )
        nsibs = len(real_names)
        if nsibs >= 3:
            phase_correct = PhaseCorrect(mat_haps=mat_haps, pat_haps=pat_haps, pos=pos)
            phase_correct.add_baf(bafs)
            # NOTE: we should use these estimates from the previous HMM runs rather than re-estimating ...
            if snakemake.params["use_prev_params"]:
                pi0_est_acc, sigma_est_acc = extract_parameters(
                    aneuploidy_df,
                    mother=snakemake.wildcards["mother"],
                    father=snakemake.wildcards["father"],
                    names=real_names,
                    chrom=c,
                )
                phase_correct.embryo_pi0s = pi0_est_acc
                phase_correct.embryo_sigmas = sigma_est_acc
            else:
                phase_correct.est_sigma_pi0s()
            (
                mat_haps,
                pat_haps,
                n_mis_mat_tot,
                n_mis_pat_tot,
            ) = phase_correct.viterbi_phase_correct(niter=2)
            pi0_ests = phase_correct.embryo_pi0s
            sigma_ests = phase_correct.embryo_sigmas
            recomb_dict[c] = {}
            # Use the least noisy siblings to help with estimation here ...
            idxs = np.argsort(sigma_ests)
            for i in range(nsibs):
                paths0 = []
                for j in idxs:
                    if j != i:
                        path_ij = hmm.viterbi_path(
                            bafs=[
                                phase_correct.embryo_bafs[i],
                                phase_correct.embryo_bafs[j],
                            ],
                            pos=phase_correct.pos,
                            mat_haps=phase_correct.mat_haps_fixed,
                            pat_haps=phase_correct.pat_haps_fixed,
                            pi0=(pi0_ests[i], pi0_ests[j]),
                            std_dev=(sigma_ests[i], sigma_ests[j]),
                            r=1e-8,
                        )
                        paths0.append(path_ij)
                        # This ensures that the largest families still have reasonable runtimes
                        if len(paths0) > 2:
                            break
                # Isolate the recombinations here ...
                mat_rec, pat_rec, _, _ = hmm.isolate_recomb(
                    paths0[0], paths0[1:], window=50
                )
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
                # NOTE: Cases of no crossover recombination detected as well ...
                if mat_rec is []:
                    lines.append(
                        f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tmaternal\t{nan}\t{nan}\t{nan}\t{pi0_ests[i]}\t{sigma_ests[i]}\n'
                    )
                if pat_rec is []:
                    lines.append(
                        f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tpaternal\t{nan}\t{nan}\t{nan}\t{pi0_ests[i]}\t{sigma_ests[i]}\n'
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
