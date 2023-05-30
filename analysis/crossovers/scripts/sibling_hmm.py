import gzip as gz
import pickle
import sys
import numpy as np
from karyohmm import QuadHMM

def prepare_paired_data(embryo_id1='10013440016_R06C01', embryo_id2='10013440016_R04C01', embryo_id3='10013440016_R05C01', chrom='chr21', data_dict=test_family_data):
    """Create the filtered dataset for evaluating crossovers using the QuadHMM results."""
    data_embryo1 = test_family_data[embryo_id1][chrom]
    data_embryo2 = test_family_data[embryo_id2][chrom]
    data_embryo3 = test_family_data[embryo_id3][chrom]
    if (data_embryo1['pos'].size != data_embryo2['pos'].size) or ((data_embryo1['pos'].size != data_embryo2['pos'].size)):
        pos1 = data_embryo1['pos']
        pos2 = data_embryo2['pos']
        pos3 = data_embryo3['pos']
        idx2 = np.isin(pos2, pos1) & np.isin(pos2, pos3)
        idx1 = np.isin(pos1, pos2) & np.isin(pos1, pos3)
        idx3 = np.isin(pos3, pos1) & np.isin(pos3, pos2)
        baf1 = data_embryo1['baf_embryo'][idx1]
        baf2 = data_embryo2['baf_embryo'][idx2]
        baf3 = data_embryo3['baf_embryo'][idx3]
        
        mat_haps = data_embryo1['mat_haps'][:,idx1]
        pat_haps = data_embryo1['pat_haps'][:,idx1]
        assert baf1.size == baf2.size
        assert baf2.size == baf3.size
        # Return the maternal haplotypes, paternal haplotypes, baf
        return mat_haps, pat_haps, baf1, baf2, baf3, pos1
    else:
        pos = data_embryo1['pos']
        baf1 = data_embryo1['baf_embryo']
        baf2 = data_embryo2['baf_embryo']
        baf3 = data_embryo3['baf_embryo']
        mat_haps = data_embryo1['mat_haps']
        pat_haps = data_embryo1['pat_haps']
        # Return the maternal haplotypes, paternal haplotypes, baf
        assert baf1.size == baf2.size
        assert baf2.size == baf3.size
        return mat_haps, pat_haps, baf1, baf2, baf3, pos

def find_nearest_het(pos, haps):
    """Find the nearest heterozygote."""
    assert pos.size == haps.shape[1]
    pass


if __name__ == "__main__":
    # Read in the input data and params ...
    eps = 10 ** snakemake.params["eps"]
    lrr = snakemake.params["lrr"]
    data = pickle.load(gz.open(snakemake.input["baf_pkl"], "r"))
    full_chrom_hmm_dict = {}
    for c in snakemake.params["chroms"]:
        baf_data = data[c]
        print("Running meta HMM parameter estimation ...", file=sys.stderr)
        n01 = np.nansum((baf_data["baf_embryo"] == 1) | (baf_data["baf_embryo"] == 0))
        m = baf_data["baf_embryo"].size
        if n01 == m:
            print("Warning: all BAF values were either [0,1].", file=sys.stderr)
            res_dict = {
                "0": np.nan,
                "1m": np.nan,
                "1p": np.nan,
                "2m": np.nan,
                "2p": np.nan,
                "2": np.nan,
                "3m": np.nan,
                "3p": np.nan,
                "sigma_baf": np.nan,
                "pi0_baf": np.nan,
                "pi0_lrr": np.nan,
                "lrr_mu": np.nan,
                "lrr_sd": np.nan,
                "aploid": baf_data["aploid"],
            }
        else:
            logr = False
            lrrs = np.ones(baf_data["baf_embryo"].size)
            if lrr == "raw":
                lrrs = np.nan_to_num(baf_data["lrr_embryo_raw"])
                logr = True
            elif lrr == "norm":
                lrrs = np.nan_to_num(baf_data["lrr_embryo_norm"])
                logr = True
            hmm = MetaHMM(logr=logr)
            # NOTE: this naively just takes every other snp to reduce runtimes ...
            pi0_est, sigma_est = hmm.est_sigma_pi0(
                bafs=baf_data["baf_embryo"][::2],
                lrrs=lrrs[::2],
                mat_haps=baf_data["mat_haps"][:, ::2],
                pat_haps=baf_data["pat_haps"][:, ::2],
                eps=eps,
                unphased=snakemake.params["unphased"],
                logr=False,
            )
            pi0_lrr = np.nan
            lrr_mu = None
            lrr_sd = None
            if logr:
                pi0_lrr, lrr_mu, lrr_sd, _ = hmm.est_lrr_sd(lrrs, niter=50)
            print("Finished meta HMM parameter estimation ...", file=sys.stderr)
            print("Starting meta HMM forward-backward algorithm", file=sys.stderr)
            gammas, states, karyotypes = hmm.forward_backward(
                bafs=baf_data["baf_embryo"],
                lrrs=lrrs,
                mat_haps=baf_data["mat_haps"],
                pat_haps=baf_data["pat_haps"],
                pi0=pi0_est,
                std_dev=sigma_est,
                pi0_lrr=pi0_lrr,
                lrr_mu=lrr_mu,
                lrr_sd=lrr_sd,
                eps=eps,
                unphased=snakemake.params["unphased"],
                logr=logr,
            )
            print(
                "Finished running meta HMM forward-backward algorithm ...",
                file=sys.stderr,
            )
            res_dict = hmm.posterior_karyotypes(gammas, karyotypes)
            res_dict["sigma_baf"] = sigma_est
            res_dict["pi0_baf"] = pi0_est
            res_dict["pi0_lrr"] = pi0_lrr
            res_dict["lrr_mu"] = (
                np.nan if lrr_mu is None else ",".join([str(m) for m in lrr_mu])
            )
            res_dict["lrr_sd"] = (
                np.nan if lrr_sd is None else ",".join([str(s) for s in lrr_sd])
            )
            res_dict["aploid"] = baf_data["aploid"]
            res_dict["karyotypes"] = karyotypes
            res_dict["gammas"] = gammas.astype(np.float16)
            res_dict["states"] = np.array(
                [hmm.get_state_str(s) for s in states], dtype=str
            )
        try:
            res_dict["mother_id"] = snakemake.params["mother_id"]
            res_dict["father_id"] = snakemake.params["father_id"]
            res_dict["child_id"] = snakemake.params["child_id"]
        except KeyError:
            pass
        full_chrom_hmm_dict[c] = res_dict
    pickle.dump(full_chrom_hmm_dict, gz.open(snakemake.output["hmm_pkl"], "wb"))
