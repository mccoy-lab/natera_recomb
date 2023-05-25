import sys

import numpy as np
from karyohmm import QuadHMM


def identify_opposite_homozygotes(rec, haps):
    """NOTE: """
    pass


if __name__ == "__main__":
    # Setup the HMM class ...
    hmm = QuadHMM()
    # Read in the input data and params ...
    baf_data = np.load(snakemake.input["baf"])
    eps = 10 ** snakemake.params["eps"]
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
    assert baf_data["nsibs"] >= 3
    # NOTE: use the erroneous haplotypes that may have switch errors here ...
    mat_haps = baf_data["mat_haps_real"]
    pat_haps = baf_data["pat_haps_real"]
    res_dict = {}
    res_dict["r"] = r
    res_dict["aploid"] = baf_data["aploid"]
    res_dict["nsibs"] = baf_data["nsibs"]
    # NOTE: lets try this now with certain settings for now?
    for i in range(1, nsibs, 2):
        rec_dict, _, path_dict = hmm.isolate_recomb_triplet(
            bafs=[
                res_dict["baf_embryo0"],
                res_dict[f"baf_embryo{i}"],
                res_dict[f"baf_embryo{i+1}"],
            ],
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            r=1e-6,
            pi0=0.3,
            std_dev=0.15,
            eps=1e-5,
        )
        if "mat_rec0" not in res_dict:
            res_dict["mat_rec0"] = rec_dict["mat_rec0"]
            res_dict["pat_rec0"] = rec_dict["pat_rec0"]
        res_dict[f"mat_rec{i}"] = rec_dict[f"mat_rec{i}"]
        res_dict[f"pat_rec{i}"] = rec_dict[f"pat_rec{i}"]
        res_dict[f"mat_rec{i+1}"] = rec_dict[f"mat_rec{i+1}"]
        res_dict[f"pat_rec{i+1}"] = rec_dict[f"pat_rec{i+1}"]
    if "pos" in baf_data:
        res_dict["pos"] = baf_data["pos"]
    # Write out the hmm results to a compressed readout ...
    np.savez_compressed(snakemake.output["hmm_out"], **res_dict)
