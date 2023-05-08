import sys

import numpy as np
from karyohmm import EuploidyHMM

if __name__ == "__main__":
    # Setup the HMM class ...
    hmm = EuploidyHMM()
    # Read in the input data and params ...
    baf_data = np.load(snakemake.input["baf"])
    eps = 10 ** snakemake.params["eps"]
    # NOTE: we are assuming that this is a euploid embryo here
    r = 1e-4
    if "r" in snakemake.params:
        r = 10 ** snakemake.params["r"]
    if "pos" in baf_data:
        pos = baf_data["pos"]
        m = pos.size
        bp_len_mb = (np.max(pos) - np.min(pos)) / 1e6
        # Set the recombination distance in Morgans
        r = (0.01 * bp_len_mb) / m
    assert "nsibs" in baf_data
    # NOTE: use the erroneous ones here ...
    mat_haps = baf_data["mat_haps_real"]
    pat_haps = baf_data["pat_haps_real"]
    res_dict = {}
    res_dict["r"] = r
    res_dict["aploid"] = baf_data["aploid"]
    res_dict["nsibs"] = baf_data["nsibs"]
    for i in range(baf_data["nsibs"]):
        pi0_est, sigma_est = hmm.est_sigma_pi0(
            bafs=baf_data[f"baf_embryo{i}"],
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            eps=eps,
            r=r,
        )
        path, states, _, _ = hmm.viterbi_algorithm(
            bafs=baf_data[f"baf_embryo{i}"],
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            eps=eps,
            pi0=pi0_est,
            std_dev=sigma_est,
            r=r,
        )
        changepts, maternal_rec, paternal_rec = hmm.assign_recomb(states, path)
        # Obtain the parameters for this specific sibling and store in numpy directory
        res_dict[f"sigma_est{i}"] = sigma_est
        res_dict[f"pi0_est{i}"] = pi0_est
        res_dict[f"states{i}"] = np.array(
            [hmm.get_state_str(s) for s in states], dtype=str
        )
        res_dict[f"changepts{i}"] = changepts
        res_dict[f"maternal_rec{i}"] = maternal_rec
        res_dict[f"paternal_rec{i}"] = paternal_rec
    if "pos" in baf_data:
        res_dict["pos"] = baf_data["pos"]
    # Write out the hmm results to a compressed readout ...
    np.savez_compressed(snakemake.output["hmm_out"], **res_dict)
