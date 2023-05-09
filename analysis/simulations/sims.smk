import numpy as np
import pandas as pd


configfile: "config.yaml"


TARGETS = []


rule all:
    input:
        TARGETS,


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
        "results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    output:
        "results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.{prop}.filtered.npz",
    wildcard_constraints:
        prop="\d+",
    params:
        prop=lambda wildcards: int(wildcards.prop) / 100,
    script:
        "scripts/filt_phase_err.py"


rule concat_true_inferred:
    """Create a table which contatenates the true and inferred results."""
    input:
        filtered="results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.{prop}.filtered.npz",
        true_co="results/sims/true_co_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    output:
        co_tsv=temp(
            "results/sims/co_compare_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.tsv"
        ),
    run:
        filt_infer_data = np.load(input.filtered)
        true_data = np.load(input.true_co)
        nsibs = int(wildcards.nsibs)
        with open(output.co_tsv, "w+") as out:
            out.write(
                "rep\tpi0\tsigma\tm\tphase_err\tnsibs\tprop\tsib_index\ttrue_co_pat\tinf_co_pat\ttrue_co_mat\tinf_co_pat\n"
            )
            for i in nsibs:
                pass
