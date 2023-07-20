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
    r = 1e-16
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
    mat_haps = baf_data["mat_haps_real"]
    pat_haps = baf_data["pat_haps_real"]
    res_dict = {}
    res_dict["r"] = r
    res_dict["aploid"] = baf_data["aploid"]
    res_dict["nsibs"] = nsibs
    # We just do this for the first embryo as a small sanity check ...  
    paths = []
    est_pi0s = []
    est_sigmas = []
    i = 0
    for j in range(1, nsibs):
        pi0_x, sigma_x = hmm.est_sigma_pi0(
                bafs=[baf_data[f"baf_embryo{i}"][::5], baf_data[f"baf_embryo{j}"][::5]], 
                mat_haps=mat_haps[:,::5], 
                pat_haps=pat_haps[:,::5],
                algo = "L-BFGS-B",
                r=r
        )
        est_pi0s.append(pi0_x)
        est_sigmas.append(sigma_x)
        path01 = hmm.map_path(
            bafs=[baf_data[f"baf_embryo{i}"], baf_data[f"baf_embryo{j}"]],
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            std_dev=sigma_x,
            pi0=pi0_x,
            r=r,
        )
        paths.append(path01)

    # Isolate the recombination events in this single embryo ... 
    mat_rec, pat_rec = hmm.isolate_recomb(paths[0], paths[1:])
    res_dict[f"mat_rec{i}"] = mat_rec
    res_dict[f"pat_rec{i}"] = pat_rec
    res_dict[f"pi0_{i}"] = est_pi0s
    res_dict[f"sigma_{i}"] = est_sigmas

    if "pos" in baf_data:
        res_dict["pos"] = baf_data["pos"]
    # Write out the hmm results to a compressed readout ...
    np.savez_compressed(snakemake.output["hmm_out"], **res_dict)
