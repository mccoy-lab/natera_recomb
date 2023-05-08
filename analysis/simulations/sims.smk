#!python3

import numpy as np
import pandas as pd


configfile: "config.yaml"


rule all:
    input:
        phase_sims=expand(
            "results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
            rep=range(config["co_sims"]["reps"]),
            pi0=config["co_sims"]["pi0"],
            std=config["co_sims"]["std_dev"],
            m=config["co_sims"]["m"],
            p=config["co_sims"]["phase_err"],
            nsibs=config["co_sims"]["nsibs"],
        ),


rule sim_siblings:
    """Simulate sibling embryo data."""
    output:
        sim="results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    wildcard_constraints:
        m="\d+",
        p="\d+",
        pi0="\d+",
        std="\d+",
    params:
        m=lambda wildcards: int(wildcards.m),
        phase_err=lambda wildcards: int(wildcards.p) / 1000,
        nsib=lambda wildcards: int(wildcards.nsibs),
        pi0=lambda wildcards: int(wildcards.pi0) / 100,
        sigma=lambda wildcards: int(wildcards.std) / 100,
        sfs=config["afs"],
        seed=lambda wildcards: int(wildcards.rep) + 1,
    script:
        "scripts/sim_siblings.py"


rule isolate_true_crossover:
    """Isolate crossovers from simulations."""
    input:
        sib_data=rules.sim_siblings.output.sim,
    output:
        true_co="results/sims/true_co_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    run:
        nsibs = int(wildcards.nsibs)
        data = np.load(input.sib_data)
        data_dict = {}
        for i in range(nsibs):
            zs_maternal = data[f"zs_maternal{i}"]
            zs_paternal = data[f"zs_paternal{i}"]
            # Shifts in indices are indicative of crossover
            co_mat_pos = np.where(zs_maternal[1:] != zs_maternal[:-1])[0]
            co_pat_pos = np.where(zs_paternal[1:] != zs_paternal[:-1])[0]
            data_dict[f"co_pos_mat_{i}"] = co_mat_pos
            data_dict[f"co_pos_pat_{i}"] = co_pat_pos
        np.savez_compressed(output.true_co, **data_dict)


rule estimate_co_hmm:
    """Estimate crossover and their locations using the viterbi algorithm."""
    input:
        baf=rules.sim_siblings.output.sim,
    output:
        hmm_out="results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    params:
        eps=-6,
    script:
        "scripts/hmm_siblings.py"


rule filter_co_phase_err:
    """Filter true crossovers from phase errors."""
    input:
        "results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    output:
        "results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.filtered.npz",
    script:
        "scripts/filt_phase_err.py"
