#!python3

import numpy as np 
import pandas as pd

rule sim_siblings:
    """Simulate sibling embryo data."""
    input:
        afs = "utils/afs_spectrum.tsv.gz"
    output:
        sim = "results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz"
    wildcard_constraints:
        m="\d+",
        p="\d+",
        pi0="\d+",
        std="\d+"
    params:
        m = lambda wildcards: int(wildcards.m),
        phase_err = lambda wildcards: int(wildcards.p)/1000,
        nsibs = lambda wildcards: int(wildcards.nsibs),
        pi0 = lambda wildcards: int(wildcards.pi0)/100,
        stddev = lambda wildcards: int(wildcards.std)/100
    script:
        "scripts/sim_siblings.py"

rule isolate_true_crossover:
    """Isolate true crossovers """
    input:
        sib_data = rules.sim_siblings.output.sim
    output:
        "results/sims/true_co_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz" 
    run:
        nsibs = int(wildcards.nsibs)
        data = np.load(input.sib_data)
        

rule estimate_co:
    """Estimate crossovers"""
    input:
        baf_data = rules.sim_siblings.output.sim
    output:
        "results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz"
    script:
        "scripts/viterbi.py"

