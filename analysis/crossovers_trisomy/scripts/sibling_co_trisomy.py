import gzip as gz
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from karyohmm import MetaHMM, PhaseCorrect
from tqdm import tqdm
from utils import *
import ruptures as rpt


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
    trisomy_df = pd.read_csv(["trisomy_calls"], sep="\t")
    baf_data = pickle.load(gz.open(snakemake.input["baf_pkl"], "rb"))
    hmm_data = pickle.load(gz.open(snakemake.input["hmm_pkl"], "rb"))
    hmm = MetaHMM()
    if call == "3m":
        hmm.states = hmm.m_trisomy_states
        hmm.karyotypes = np.array(["3m", "3m", "3m", "3m", "3m", "3m"], dtype=str)
    elif call == "3p":
        hmm.states = hmm.p_trisomy_states
        hmm.karyotypes = np.array(["3p", "3p", "3p", "3p", "3p", "3p"], dtype=str)
    else:
        raise ValueError("Chromosome is not determined to be a trisomy!")
    gammas, states, karyotypes = hmm.forward_backward(
        bafs=bafs,
        pos=pos,
        mat_haps=mat_haps,
        pat_haps=pat_haps,
        pi0=pi0_est,
        std_dev=sigma_est,
        unphased=False,
    )

    pickle.dump(recomb_dict, gz.open(snakemake.output["recomb_paths"], "wb"))
    # Write out the formal crossover spot output here
    with open(snakemake.output["est_recomb"], "w") as out:
        out.write(
            "mother\tfather\tchild\tchrom\tcrossover_sex\tmin_pos\tavg_pos\tmax_pos\tavg_pi0\tavg_sigma\n"
        )
        for line in lines:
            out.write(line)
