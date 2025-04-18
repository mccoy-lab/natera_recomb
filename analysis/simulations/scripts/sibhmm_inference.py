import sys

import numpy as np
from karyohmm import QuadHMM


def nearest_het(rec, haps):
    """NOTE: this defines the bounds of heterozygotes in the specific parent."""
    assert haps.ndim == 2
    assert rec >= 0 and rec < haps.shape[1]
    assert np.all(np.isin(haps, [0, 1, 2]))
    geno = haps[0, :] + haps[1, :]
    het_idx = np.where(geno == 1)[0]
    low_idx = np.max(het_idx < rec)
    high_idx = np.min(het_idx > rec)
    return low_idx, high_idx


def isolate_bounds(recs, haps):
    """Isolating the bounds of the recombination events here."""
    bounds = []
    for r in recs:
        l, h = nearest_het(r, haps)
        bounds.append([l, h])
    return bounds


if __name__ == "__main__":
    # Setup the HMM class ...
    hmm = QuadHMM()
    # Read in the input data and params ...
    baf_data = np.load(snakemake.input["baf"])
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
    nsibs = baf_data["nsibs"]
    # NOTE: use the erroneous haplotypes that may have switch errors here ...
    if snakemake.params["corrected"]:
        mat_haps = baf_data["mat_haps_fixed"]
        pat_haps = baf_data["pat_haps_fixed"]
    else:
        mat_haps = baf_data["mat_haps_real"]
        pat_haps = baf_data["pat_haps_real"]
    est_pi0s = baf_data["est_pi0s"]
    est_sigmas = baf_data["est_sigmas"]
    res_dict = {}
    res_dict["r"] = r
    res_dict["aploid"] = baf_data["aploid"]
    res_dict["nsibs"] = nsibs
    # We just do this for the first embryo as a small sanity check ...
    paths = []
    paths_true = []
    i = 0
    for j in range(1, nsibs):
        # NOTE: we don't employ the parameter inference step here but assume that it has been done slightly earlier in a disomy model
        pi0_x = np.median([est_pi0s[i], est_pi0s[j]])
        sigma_x = np.median([est_sigmas[i], est_sigmas[j]])
        path01 = hmm.map_path(
            bafs=[baf_data[f"baf_embryo{i}"], baf_data[f"baf_embryo{j}"]],
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            std_dev=sigma_x,
            pi0=pi0_x,
            r=r,
        )
        paths.append(path01)
        path01_true = hmm.map_path(
            bafs=[baf_data[f"baf_embryo{i}"], baf_data[f"baf_embryo{j}"]],
            mat_haps=baf_data["mat_haps_true"],
            pat_haps=baf_data["pat_haps_true"],
            std_dev=sigma_x,
            pi0=pi0_x,
            r=r,
        )
        paths_true.append(path01_true)

    # Isolate the recombination events in this single embryo ...
    mat_rec, pat_rec, mat_rec_dict, pat_rec_dict = hmm.isolate_recomb(
        paths[0], paths[1:]
    )
    mat_rec_true, pat_rec_true, mat_rec_dict, pat_rec_dict = hmm.isolate_recomb(
        paths_true[0], paths_true[1:]
    )
    res_dict[f"mat_rec{i}"] = mat_rec
    res_dict[f"pat_rec{i}"] = pat_rec
    res_dict[f"mat_rec_truehap{i}"] = mat_rec_true
    res_dict[f"pat_rec_truehap{i}"] = pat_rec_true
    res_dict[f"mat_rec_dict{i}"] = mat_rec_dict
    res_dict[f"pat_rec_dict{i}"] = pat_rec_dict

    res_dict[f"pi0_{i}"] = est_pi0s
    res_dict[f"sigma_{i}"] = est_sigmas

    if "pos" in baf_data:
        res_dict["pos"] = baf_data["pos"]
    # Write out the hmm results to a compressed readout ...
    np.savez_compressed(snakemake.output["hmm_out"], **res_dict)
