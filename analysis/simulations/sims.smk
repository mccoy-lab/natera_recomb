import numpy as np
import pandas as pd


configfile: "config.yaml"


rule all:
    input:
        "results/sims/total_co_sims.tsv",


rule sim_siblings:
    """Simulate sibling embryo data."""
    output:
        sim="results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    wildcard_constraints:
        m="\d+",
        p="\d+",
        pi0="\d+",
        std="\d+",
        nsibs="\d+",
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
        sib_data="results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
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
            data_dict[f"cos_pos_mat_{i}"] = co_mat_pos
            data_dict[f"cos_pos_pat_{i}"] = co_pat_pos
        np.savez_compressed(output.true_co, **data_dict)


rule estimate_co_hmm:
    """Estimate crossover and their locations using the viterbi algorithm."""
    input:
        baf="results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    output:
        hmm_out="results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    params:
        eps=-6,
    script:
        "scripts/hmm_siblings.py"


rule filter_co_phase_err:
    """Filter true crossovers from phase errors."""
    input:
        infer_hmm_co="results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    output:
        filt_co="results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.{prop}.filtered.npz",
    params:
        nsibs=lambda wildcards: int(wildcards.nsibs),
        prop=lambda wildcards: int(wildcards.prop) / 100,
    script:
        "scripts/filter_phase_err.py"


rule concat_true_inferred:
    """Create a table which contatenates the true and inferred results."""
    input:
        filtered="results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.{prop}.filtered.npz",
        true_co="results/sims/true_co_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    output:
        co_tsv=temp(
            "results/sims/co_compare_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.{prop}.tsv"
        ),
    wildcard_constraints:
        rep="\d+",
        pi0="\d+",
        std="\d+",
        m="\d+",
        nsibs="\d+",
        p="\d+",
        prop="\d+",
    run:
        filt_infer_data = np.load(input.filtered)
        true_data = np.load(input.true_co)
        nsibs = int(wildcards.nsibs)
        rep = int(wildcards.rep)
        pi0 = int(wildcards.pi0) / 100
        std = int(wildcards.std) / 100
        m = int(wildcards.m)
        phase_err = int(wildcards.p) / 1000
        prop = int(wildcards.prop) / 100
        with open(output.co_tsv, "w+") as out:
            out.write(
                "rep\tpi0\tsigma\tm\tphase_err\tnsibs\tprop\tsib_index\ttrue_co_mat\tinf_co_mat\ttrue_co_pat\tinf_co_pat\n"
            )
            for i in range(nsibs):
                cos_pos_mat = ",".join([str(x) for x in true_data[f"cos_pos_mat_{i}"]])
                cos_pos_pat = ",".join([str(x) for x in true_data[f"cos_pos_pat_{i}"]])
                mat_rec_filt = ",".join(
                    [str(x) for x in filt_infer_data[f"mat_rec_filt_{i}"]]
                )
                pat_rec_filt = ",".join(
                    [str(x) for x in filt_infer_data[f"pat_rec_filt_{i}"]]
                )
                out.write(
                    f"{rep}\t{pi0}\t{std}\t{m}\t{phase_err}\t{nsibs}\t{prop}\t{i}\t{cos_pos_mat}\t{mat_rec_filt}\t{cos_pos_pat}\t{pat_rec_filt}\n"
                )


rule collect_crossover_results:
    """Collection rule for the full set of simulations."""
    input:
        co_res=expand(
            "results/sims/co_compare_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.{prop}.tsv",
            rep=range(config["co_sims"]["reps"]),
            pi0=config["co_sims"]["pi0"],
            std=config["co_sims"]["std_dev"],
            m=config["co_sims"]["m"],
            p=config["co_sims"]["phase_err"],
            nsibs=config["co_sims"]["nsibs"],
            prop=80,
        ),
    output:
        tsv="results/sims/total_co_sims.tsv",
    run:
        dfs = []
        for fp in input.co_res:
            dfs.append(pd.read_csv(fp, sep="\t"))
        tot_df = pd.concat(dfs)
        tot_df.to_csv(output.tsv, sep="\t", index=None)
