import sys

import numpy as np
import pandas as pd
from karyohmm import PhaseCorrect


if __name__ == "__main__":
    # Read in the input data and params ...
    baf_data = np.load(snakemake.input["sim"])
    assert "pos" in baf_data
    pos = baf_data["pos"]
    assert "nsibs" in baf_data
    assert baf_data["nsibs"] >= 2
    nsibs = baf_data["nsibs"]
    mat_haps = baf_data["mat_haps_real"]
    pat_haps = baf_data["pat_haps_real"]
    phase_correct = PhaseCorrect(
        mat_haps=baf_data["mat_haps_real"], pat_haps=baf_data["pat_haps_real"], pos=pos
    )
    phase_correct.add_true_haps(
        true_mat_haps=baf_data["mat_haps_true"], true_pat_haps=baf_data["pat_haps_true"]
    )
    phase_correct.add_baf(
        embryo_bafs=[baf_data[f"baf_embryo{i}"] for i in range(nsibs)]
    )
    phase_correct.est_sigma_pi0s()
    acc = []
    seed = (
        snakemake.params["seed"]
        + int(snakemake.params["phase_err"] * 1e3)
        + int(snakemake.params["nsib"] * 100)
    )
    # Run the full simulation using the defined helper function
    # NOTE: This now should have an expected number of one crossover per chromosome per parent
    m = snakemake.params["m"]
    nsibs = snakemake.params["nsib"]
    std_dev = snakemake.params["sigma"]
    mix_prop = snakemake.params["pi0"]
    switch_err_rate = snakemake.params["phase_err"]

    # Run the phase_error checking
    rs = [1e-2, 1e-4, 1e-8]
    for r in rs:
        phase_correct.viterbi_phase_correct(r=r)
        _, _, se_m, _, _, _ = phase_correct.estimate_switch_err_true()
        _, _, se_mr, _, _, _ = phase_correct.estimate_switch_err_true(fixed=True)
        _, _, se_p, _, _, _ = phase_correct.estimate_switch_err_true(maternal=False)
        _, _, se_pr, _, _, _ = phase_correct.estimate_switch_err_true(
            maternal=False, fixed=True
        )
        acc.append(
            [
                m,
                nsibs,
                std_dev,
                mix_prop,
                switch_err_rate,
                se_m,
                se_mr,
                se_p,
                se_pr,
                r,
                np.nan,
                "Viterbi",
                seed,
            ]
        )
    lods = [-1.0, -2.0, -5.0]
    for l in lods:
        phase_correct.lod_phase_correct(
            lod_thresh=l,
            pi0=np.median(phase_correct.embryo_pi0s),
            std_dev=np.median(phase_correct.embryo_sigmas),
        )
        # Phase correct the paternal haplotypes using median parameter inferred across all siblings
        phase_correct.lod_phase_correct(
            maternal=False,
            lod_thresh=l,
            pi0=np.median(phase_correct.embryo_pi0s),
            std_dev=np.median(phase_correct.embryo_sigmas),
        )
        _, _, se_m, _, _, _ = phase_correct.estimate_switch_err_true()
        _, _, se_mr, _, _, _ = phase_correct.estimate_switch_err_true(fixed=True)
        _, _, se_p, _, _, _ = phase_correct.estimate_switch_err_true(maternal=False)
        _, _, se_pr, _, _, _ = phase_correct.estimate_switch_err_true(
            maternal=False, fixed=True
        )
        acc.append(
            [
                m,
                nsibs,
                std_dev,
                mix_prop,
                switch_err_rate,
                se_m,
                se_mr,
                se_p,
                se_pr,
                np.nan,
                l,
                "LOD Score",
                seed,
            ]
        )

    df = pd.DataFrame(acc)
    df.columns = [
        "n_snps",
        "n_sibs",
        "sigma",
        "pi0",
        "switch_err",
        "switch_err_emp_maternal",
        "switch_err_emp_maternal_fixed",
        "switch_err_emp_paternal",
        "switch_err_emp_paternal_fixed",
        "recomb_param",
        "lod_threshold",
        "method",
        "seed",
    ]
    df.to_csv(snakemake.output["tsv"], sep="\t", index=None)
