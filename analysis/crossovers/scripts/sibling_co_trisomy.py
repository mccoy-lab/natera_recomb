import gzip as gz
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from karyohmm import MetaHMM, PhaseCorrect
from tqdm import tqdm
from utils import *


if __name__ == "__main__":
    # Read in the input data and params ...
    trisomy_df = pd.read_csv(["trisomy_calls"], sep="\t")
    baf_data = np.load(gz.open(snakemake.input["baf_pkl"]), allow_pickle=True)
    hmm = MetaHMM()
    if call == '3m':
        hmm.states = hmm.m_trisomy_states
        hmm.karyotypes = np.array(['3m', '3m', '3m', '3m', '3m', '3m'], dtype=str)
    elif call == '3p':
        hmm.states = hmm.p_trisomy_states
        hmm.karyotypes = np.array(['3p', '3p', '3p', '3p', '3p', '3p'], dtype=str)
    else:
        raise ValueError('Chromosome is not determined to be a trisomy!')
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=bafs, pos=pos, mat_haps=mat_haps, pat_haps=pat_haps
    )
    # NOTE: should we do a viterbi here instead?
    gammas, states, karyotypes = hmm.forward_backward(
        bafs=bafs,
        pos=pos,
        mat_haps=mat_haps,
        pat_haps=pat_haps,
        pi0=pi0_est,
        std_dev=sigma_est,
        unphased=True,
    )
    
    pickle.dump(recomb_dict, gz.open(snakemake.output["recomb_paths"], "wb"))
    # Write out the formal crossover spot output here
    with open(snakemake.output["est_recomb"], "w") as out:
        out.write(
            "mother\tfather\tchild\tchrom\tcrossover_sex\tmin_pos\tavg_pos\tmax_pos\tavg_pi0\tavg_sigma\n"
        )
        for line in lines:
            out.write(line)
