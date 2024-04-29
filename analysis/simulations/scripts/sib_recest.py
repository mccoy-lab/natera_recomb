import sys

import numpy as np
from karyohmm import RecombEst, MetaHMM

if __name__ == "__main__":
    # Setup the HMM class ...
    hmm = MetaHMM(disomy=True)
    # Read in the input data and params ...
    baf_data = np.load(snakemake.input["baf"])
    r = 1e-8
    if "r" in snakemake.params:
        r = 10 ** snakemake.params["r"]
    assert "pos" in baf_data
    pos = baf_data["pos"]
    m = pos.size
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
    recomb_est = RecombEst(mat_haps=mat_haps, pat_haps=pat_haps, pos=pos)
    recomb_est.embryo_pi0s = est_pi0s
    recomb_est.embryo_sigmas = est_sigmas
    recomb_est_true = RecombEst(
        mat_haps=baf_data["mat_haps_true"], pat_haps=baf_data["pat_haps_true"], pos=pos
    )
    recomb_est_true.embryo_pi0s = est_pi0s
    recomb_est_true.embryo_sigmas = est_sigmas

    res_dict = {}
    res_dict["r"] = r
    res_dict["aploid"] = baf_data["aploid"]
    res_dict["nsibs"] = nsibs
    expected_baf = []
    for i in range(nsibs):
        dosages = hmm.genotype_embryo(
            bafs=baf_data[f"baf_embryo{i}"],
            pos=recomb_est.pos,
            mat_haps=recomb_est.mat_haps,
            pat_haps=recomb_est.pat_haps,
            std_dev=recomb_est.embryo_sigmas[i],
            pi0=recomb_est.embryo_pi0s[i],
        )
        e_baf_i = (dosages[0, :] * 0 + dosages[1, :] * 1 + dosages[2, :] * 2.0) / 2.0
        expected_baf.append(e_baf_i)
    recomb_est.add_baf(embryo_bafs=expected_baf)
    for i in range(nsibs):
        dosages = hmm.genotype_embryo(
            bafs=baf_data[f"baf_embryo{i}"],
            pos=recomb_est.pos,
            mat_haps=recomb_est_true.mat_haps,
            pat_haps=recomb_est_true.pat_haps,
            std_dev=recomb_est_true.embryo_sigmas[i],
            pi0=recomb_est_true.embryo_pi0s[i],
        )

        e_baf_i = (dosages[0, :] * 0 + dosages[1, :] * 1 + dosages[2, :] * 2.0) / 2.0
        expected_baf.append(e_baf_i)
    recomb_est_true.add_baf(embryo_bafs=expected_baf)
    for i in range(nsibs):
        # Isolate the recombinations here ...
        mat_rec, mat_rec_support = recomb_est.estimate_crossovers(
            template_embryo=i, maternal=True
        )
        pat_rec, pat_rec_support = recomb_est.estimate_crossovers(
            template_embryo=i, maternal=False
        )
        # Isolate the recombinations using the true haplotypes here ...
        mat_rec_true, mat_rec_support = recomb_est.estimate_crossovers(
            template_embryo=i, maternal=True
        )
        pat_rec_true, pat_rec_support = recomb_est.estimate_crossovers(
            template_embryo=i, maternal=False
        )
        res_dict[f"mat_rec{i}"] = mat_rec
        res_dict[f"pat_rec{i}"] = pat_rec
        res_dict[f"mat_rec_truehap{i}"] = mat_rec
        res_dict[f"pat_rec_truehap{i}"] = pat_rec

        res_dict[f"mat_rec_support{i}"] = mat_rec_support
        res_dict[f"pat_rec_support{i}"] = pat_rec_support
        res_dict[f"pi0_{i}"] = recomb_est.embryo_pi0s[i]
        res_dict[f"sigma_{i}"] = recomb_est.embryo_sigmas[i]
    res_dict["pos"] = baf_data["pos"]
    # Write out the hmm results to a compressed readout ...
    np.savez_compressed(snakemake.output["infer_co"], **res_dict)
