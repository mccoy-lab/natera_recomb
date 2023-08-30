import sys

import numpy as np
from karyohmm import MetaHMM, PhaseCorrect

if __name__ == "__main__":
    # Setup the disomy-HMM class only ...
    hmm = MetaHMM(disomy=True)
    # Read in the input data and params ...
    baf_data = np.load(snakemake.input["sim"])
    r = 1e-8
    if "r" in snakemake.params:
        r = 10 ** snakemake.params["r"]
    if "pos" in baf_data:
        pos = baf_data["pos"]
        m = pos.size
        bp_len_mb = (np.max(pos) - np.min(pos)) / 1e6
        r = (0.01 * bp_len_mb) / m
    assert "nsibs" in baf_data
    assert baf_data["nsibs"] >= 3
    nsibs = baf_data["nsibs"]
    # NOTE: use the erroneous haplotypes that may have switch errors here ...
    mat_haps = baf_data["mat_haps_real"]
    pat_haps = baf_data["pat_haps_real"]
    # Copy over the main things to a dictionary
    res_data = {}
    for k in baf_data.keys():
        res_data[k] = baf_data[k]
    est_pi0s = []
    est_sigmas = []
    for i in range(nsibs):
        pi0_x, sigma_x = hmm.est_sigma_pi0(
            bafs=baf_data[f"baf_embryo{i}"][::2],
            mat_haps=mat_haps[:, ::2],
            pat_haps=pat_haps[:, ::2],
            algo="Powell",
            r=r,
        )
        est_pi0s.append(pi0_x)
        est_sigmas.append(sigma_x)
    phase_correct = PhaseCorrect(mat_haps=mat_haps, pat_haps=pat_haps)
    phase_correct.add_baf(
        embryo_bafs=[baf_data[f"baf_embryo{i}"] for i in range(nsibs)]
    )
    # NOTE: Should we take the mean or average here?
    # Phase correct the maternal haplotypes using the median parameter inferred across all siblings
    phase_correct.phase_correct(
        lod_thresh=snakemake.params["log_prob"],
        pi0=np.median(est_pi0s),
        std_dev=np.median(est_sigmas),
    )
    # Phase correct the paternal haplotypes using median parameter inferred across all siblings
    phase_correct.phase_correct(
        maternal=False,
        lod_thresh=snakemake.params["log_prob"],
        pi0=np.median(est_pi0s),
        std_dev=np.median(est_sigmas),
    )
    res_data["mat_haps_fixed"] = phase_correct.mat_haps_fixed
    res_data["pat_haps_fixed"] = phase_correct.pat_haps_fixed
    res_data["est_pi0s"] = est_pi0s
    res_data["est_sigmas"] = est_sigmas

    # Write out the hmm results to a compressed readout ...
    np.savez_compressed(snakemake.output["phase_correct"], **res_data)
