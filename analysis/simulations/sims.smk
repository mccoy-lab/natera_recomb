#!python3
import numpy as np
import pandas as pd


configfile: "config.yaml"


rule all:
    input:
        "results/sims/total_co_sims.tsv.gz",


rule sim_siblings:
    """Simulate sibling embryo data.
    NOTE: we arbitrarily multiply the variables for the seed so that they are more random.
    """
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
        seed=lambda wildcards: int(wildcards.rep)
        + int(wildcards.pi0) * 3
        + int(wildcards.std) * 5,
    script:
        "scripts/sim_siblings.py"


rule correct_parental_phase:
    """Apply phase correction for parental genotypes.

    NOTE: We tune this to minimize false-positive switches specifically ...
    """
    input:
        sim="results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    output:
        phase_correct="results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.phase_correct.npz",
    wildcard_constraints:
        m="\d+",
        p="\d+",
        pi0="\d+",
        std="\d+",
        nsibs="\d+",
    params:
        r=-4,
        log_prob=np.log(0.2),
    script:
        "scripts/phase_correct.py"


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
            co_mat_pos = np.where(zs_maternal[:-1] != zs_maternal[1:])[0]
            co_pat_pos = np.where(zs_paternal[:-1] != zs_paternal[1:])[0]
            data_dict[f"cos_pos_mat_{i}"] = co_mat_pos
            data_dict[f"cos_pos_pat_{i}"] = co_pat_pos
        np.savez_compressed(output.true_co, **data_dict)


rule estimate_co_hmm:
    """Estimate crossover and their locations using the viterbi algorithm."""
    input:
        baf="results/sims/sim_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.phase_correct.npz",
    output:
        hmm_out="results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.corr{corr}.npz",
    wildcard_constraints:
        corr="0|1",
    params:
        r=-8,
        corrected=lambda wildcards: wildcards.corr == 1,
    script:
        "scripts/sibhmm_inference.py"


rule concat_true_inferred:
    """Create a table which contatenates the true and inferred results."""
    input:
        infer_co="results/sims/inferhmm_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.corr{corr}.npz",
        true_co="results/sims/true_co_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.npz",
    output:
        co_tsv=temp(
            "results/sims/co_compare_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.corr{corr}.tsv"
        ),
    wildcard_constraints:
        rep="\d+",
        pi0="\d+",
        std="\d+",
        m="\d+",
        nsibs="\d+",
        p="\d+",
        prop="\d+",
        corr="0|1",
    run:
        filt_infer_data = np.load(input.infer_co)
        true_data = np.load(input.true_co)
        nsibs = int(wildcards.nsibs)
        rep = int(wildcards.rep)
        pi0 = int(wildcards.pi0) / 100
        std = int(wildcards.std) / 100
        m = int(wildcards.m)
        phase_err = int(wildcards.p) / 1000
        r_hat = filt_infer_data["r"]
        corr = int(wildcards.corr) == 1
        with open(output.co_tsv, "w+") as out:
            out.write(
                "rep\tpi0\tsigma\tm\tphase_err\tcorrected\tnsibs\tsib_index\tr_hat\tpi0_sib\tsigma_sib\ttrue_co_mat\tinf_co_mat\tinf_co_mat_truehap\ttrue_co_pat\tinf_co_pat\tinf_co_pat_truehap\n"
            )
            i = 0
            cos_pos_mat = ",".join([str(x) for x in true_data[f"cos_pos_mat_{i}"]])
            cos_pos_pat = ",".join([str(x) for x in true_data[f"cos_pos_pat_{i}"]])
            mat_rec_filt = ",".join([str(x) for x in filt_infer_data[f"mat_rec{i}"]])
            pat_rec_filt = ",".join([str(x) for x in filt_infer_data[f"pat_rec{i}"]])
            mat_rec_truehap = ",".join([str(x) for x in filt_infer_data[f"mat_rec_truehap{i}"]])
            pat_rec_truehap = ",".join([str(x) for x in filt_infer_data[f"pat_rec_truehap{i}"]])
            pi0_sib = np.mean(filt_infer_data[f"pi0_{i}"])
            sigma_sib = np.mean(filt_infer_data[f"sigma_{i}"])
            out.write(
                f"{rep}\t{pi0}\t{std}\t{m}\t{phase_err}\t{corr}\t{nsibs}\t{i}\t{r_hat}\t{pi0_sib}\t{sigma_sib}\t{cos_pos_mat}\t{mat_rec_filt}\t{mat_rec_truehap}\t{cos_pos_pat}\t{pat_rec_filt}\t{pat_rec_truehap}\n"
            )


rule collect_crossover_results:
    """Collection rule for the full set of simulations."""
    input:
        co_res=expand(
            "results/sims/co_compare_{rep}.pi0_{pi0}.std_{std}.m{m}.phase_err{p}.{nsibs}.corr{corr}.tsv",
            rep=range(config["co_sims"]["reps"]),
            pi0=config["co_sims"]["pi0"],
            std=config["co_sims"]["std_dev"],
            m=config["co_sims"]["m"],
            p=config["co_sims"]["phase_err"],
            nsibs=config["co_sims"]["nsibs"],
            corr=["0", "1"],
        ),
    output:
        tsv="results/sims/total_co_sims.tsv.gz",
    run:
        dfs = []
        for fp in input.co_res:
            dfs.append(pd.read_csv(fp, sep="\t"))
        tot_df = pd.concat(dfs)
        tot_df.to_csv(output.tsv, sep="\t", index=None)
