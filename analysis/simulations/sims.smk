#!python3

import numpy as np
import pandas as pd


configfile: "config.yaml"


rule all:
    input:
        phase_sims=expand(
            "results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
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
    """Isolate true crossovers from the simulations."""
    input:
        sib_data=rules.sim_siblings.output.sim,
    output:
        "results/sims/true_co_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    run:
        nsibs = int(wildcards.nsibs)
        data = np.load(input.sib_data)


rule estimate_co_hmm:
    """Estimate crossovers"""
    input:
        baf_data=rules.sim_siblings.output.sim,
    output:
        hmm_out="results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    script:
        "scripts/hmm_siblings.py"
