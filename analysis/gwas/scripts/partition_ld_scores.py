import numpy as np
import pandas as pd


def partition_ld_scores(ldscores, nq=4, qx=1):
    """Partition LD scores by quantiles."""
    bins = np.quantile(ldscores, q=np.linspace(0, 1, nq + 1))
    idx = np.digitize(ldscores, bins=bins)
    if qx == nq:
        return np.where(np.isin(idx, [qx, qx + 1]))[0]
    else:
        return np.where(idx == qx)[0]


def maf_partition(mafs, maf_bins=[0.0, 0.01, 0.05, 0.1, 1.0], bx=1):
    """MAF partitioning into bins."""
    idx = np.digitize(mafs, bins=maf_bins)
    return np.where(idx == bx)[0]


if __name__ == "__main__":
    # read in the full dataset
    ldscore_df = pd.concat([pd.read_csv(fp, sep="\s+") for fp in snakemake.input])
    ldscore_df["maf"] = np.minimum(ldscore_df["freq"], 1.0 - ldscore_df["freq"])

    # Get indexes of LD-score partitioned SNPs
    ld_idx = partition_ld_scores(
        ldscore_df["ldscore_SNP"].values,
        nq=snakemake.params["nld_bins"],
        qx=int(snakemake.wildcards["p"]) + 1,
    )
    filt_ldscore_df = ldscore_df.iloc[ld_idx]
    # Get indexes of MAF-partitioned SNPs as well
    maf_idx = maf_partition(
        filt_ldscore_df["maf"].values,
        maf_bins=snakemake.params["maf_bins"],
        bx=int(snakemake.wildcards["i"]) + 1,
    )
    filt_ld_mafscore_df = filt_ldscore_df.iloc[maf_idx]
    filt_ld_mafscore_df["SNP"].to_csv(
        snakemake.output["ld_maf_partition"], index=None, sep="\t", header=None
    )
