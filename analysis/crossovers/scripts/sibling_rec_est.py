import gzip as gz
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from karyohmm import MetaHMM, PhaseCorrect, RecombEst
from tqdm import tqdm
from utils import *

if __name__ == "__main__":
    # Read in the input data and params ...
    aneuploidy_df = pd.read_csv(snakemake.input["aneuploidy_calls"], sep="\t")
    hmm_dis = MetaHMM(disomy=True)
    family_data = load_baf_data(snakemake.input["baf_pkl"])
    hmm_data = load_baf_data(snakemake.input["hmm_pkl"])
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
        if pos is not None:
            posterior_disomy, posterior_pos = est_genotype_quality(
                hmm_data, names=real_names, chrom=c
            )
            assert posterior_pos.size == pos.size
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
            # Actually run the phase-correction routine
            if snakemake.params["phaseCorrect"]:
                phase_correct.viterbi_phase_correct(niter=1)
            else:
                phase_correct.mat_haps_fixed = phase_correct.mat_haps
                phase_correct.pat_haps_fixed = phase_correct.pat_haps
            pi0_ests = phase_correct.embryo_pi0s
            sigma_ests = phase_correct.embryo_sigmas
            recomb_est = RecombEst(
                mat_haps=phase_correct.mat_haps_fixed,
                pat_haps=phase_correct.pat_haps_fixed,
                pos=pos,
            )
            # Set the parameters + calculate the expected BAF
            recomb_est.embryo_pi0s = pi0_ests
            recomb_est.embryo_sigmas = sigma_ests
            expected_baf = []
            for i in range(nsibs):
                dosages = hmm_dis.genotype_embryo(
                    bafs=phase_correct.embryo_bafs[i],
                    pos=recomb_est.pos,
                    mat_haps=recomb_est.mat_haps,
                    pat_haps=recomb_est.pat_haps,
                    std_dev=recomb_est.embryo_sigmas[i],
                    pi0=recomb_est.embryo_pi0s[i],
                )
                # Calculate the expected BAF for that variant in that embryo.
                e_baf_i = (
                    dosages[0, :] * 0.0 + dosages[1, :] * 1.0 + dosages[2, :] * 2.0
                ) / 2.0
                expected_baf.append(e_baf_i)
            recomb_est.add_baf(embryo_bafs=expected_baf)
            for i in range(nsibs):
                # Isolate the recombinations here ...
                mat_rec, mat_rec_support = recomb_est.estimate_crossovers(
                    template_embryo=i, maternal=True
                )
                pat_rec, pat_rec_support = recomb_est.estimate_crossovers(
                    template_embryo=i, maternal=False
                )
                for j, m in enumerate(mat_rec):
                    left_pos, right_pos = m
                    # Take the midpoint here ...
                    rec_pos = int((left_pos + right_pos) / 2)
                    geno_qual_left = posterior_disomy[pos == left_pos][0]
                    geno_qual_right = posterior_disomy[pos == right_pos][0]
                    lines.append(
                        f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tmaternal\t{left_pos}\t{rec_pos}\t{right_pos}\t{geno_qual_left}\t{geno_qual_right}\t{pi0_ests[i]}\t{sigma_ests[i]}\t{nsibs}\t{mat_rec_support[j]}\t{pos.size}\n'
                    )
                for j, p in enumerate(pat_rec):
                    left_pos, right_pos = p
                    rec_pos = int((left_pos + right_pos) / 2)
                    geno_qual_left = posterior_disomy[pos == left_pos][0]
                    geno_qual_right = posterior_disomy[pos == right_pos][0]
                    lines.append(
                        f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tpaternal\t{left_pos}\t{rec_pos}\t{right_pos}\t{geno_qual_left}\t{geno_qual_right}\t{pi0_ests[i]}\t{sigma_ests[i]}\t{nsibs}\t{pat_rec_support[j]}\t{pos.size}\n'
                    )
                # NOTE: Cases of no crossover recombination detected as well ...
                if not mat_rec:
                    lines.append(
                        f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tmaternal\tNA\tNA\tNA\tNA\tNA\t{pi0_ests[i]}\t{sigma_ests[i]}\t{nsibs}\t0\t{pos.size}\n'
                    )
                if not pat_rec:
                    lines.append(
                        f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tpaternal\tNA\tNA\tNA\tNA\tNA\t{pi0_ests[i]}\t{sigma_ests[i]}\t{nsibs}\t0\t{pos.size}\n'
                    )
        else:
            for i in range(nsibs):
                lines.append(
                    f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tmaternal\tNA\tNA\tNA\tNA\tNA\t{pi0_ests[i]}\t{sigma_ests[i]}\t{nsibs}\t0\t{pos.size}\n'
                )
                lines.append(
                    f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{real_names[i]}\t{c}\tpaternal\tNA\tNA\tNA\tNA\tNA\t{pi0_ests[i]}\t{sigma_ests[i]}\t{nsibs}\t0\t{pos.size}\n'
                )
    # Write out crossover location output here ...
    with open(snakemake.output["est_recomb"], "w") as out:
        out.write(
            "mother\tfather\tchild\tchrom\tcrossover_sex\tmin_pos\tavg_pos\tmax_pos\tmin_pos_qual\tmax_pos_qual\tavg_pi0\tavg_sigma\tnsibs\tnsib_support\tnsnps\n"
        )
        for line in lines:
            out.write(line)
