#!python3

from functools import reduce

import numpy as np
import pandas as pd
import pyBigWig
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm


# NOTE: need to be divided by the total length of the chromosome ...
def centromere_dist(chrom, pos, centromere_dict):
    pts = np.array([centromere_dict["start"][chrom], centromere_dict["end"][chrom]])
    dist = np.min(np.abs(pos - pts))
    return dist


def telomere_dist(chrom, pos, telomere_dict):
    pts = np.array([telomere_dict["start"][chrom], telomere_dict["end"][chrom]])
    dist = np.min(np.abs(pos - pts))
    return dist


def rt_dict(rt_df):
    """Generating chromosome-specific interpolation functions for replication timing."""
    assert "chrom" in rt_df.columns
    assert "midpt" in rt_df.columns
    assert "rt" in rt_df.columns
    rt_func_dict = {}
    for chrom in np.unique(rt_df.chrom.values):
        rt_func_dict[chrom] = interp1d(
            rt_df[rt_df.chrom == chrom].midpt.values,
            rt_df[rt_df.chrom == chrom].rt.values,
            bounds_error=False,
        )
    return rt_func_dict


def avg_dist_centromere(df, centromere_df, frac_siblings=0.5):
    """Compute the average distance in bp from centromere of the nearest crossover."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert "chrom" in df.columns
    assert "avg_pos" in df.columns
    assert "nsibs" in df.columns
    assert "nsib_support" in df.columns
    assert frac_siblings >= 0.5
    df["frac_siblings"] = df["nsib_support"] / (df["nsibs"] - 1)
    centromere_dict = (
        centromere_df.groupby("chrom").agg({"start": "min", "end": "max"}).to_dict()
    )
    centromere_dists = np.zeros(df.shape[0])
    for i, (chrom, pos) in tqdm(enumerate(zip(df.chrom.values, df.avg_pos.values))):
        centromere_dists[i] = centromere_dist(
            chrom, pos, centromere_dict=centromere_dict
        )
    df["centromere_dist"] = centromere_dists
    filt_df = (
        df[df["frac_siblings"] > frac_siblings]
        .groupby(["mother", "father", "child", "chrom", "crossover_sex"])
        .agg({"centromere_dist": "min"})
        .reset_index()
        .groupby(["mother", "father", "crossover_sex"])
        .agg({"centromere_dist": "mean"})
        .reset_index()
    )
    mother_df = filt_df[filt_df.crossover_sex == "maternal"][
        ["mother", "mother", "centromere_dist"]
    ]
    mother_df.columns = ["FID", "IID", "CentromereDist"]
    father_df = filt_df[filt_df.crossover_sex == "paternal"][
        ["father", "father", "centromere_dist"]
    ]
    father_df.columns = ["FID", "IID", "CentromereDist"]
    tot_df = pd.concat([mother_df, father_df])
    return tot_df


def avg_dist_telomere(df, telomere_df, frac_siblings=0.5):
    """Compute the average distance in bp from the telomeres of the nearest crossover."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert "chrom" in df.columns
    assert "avg_pos" in df.columns
    assert "nsibs" in df.columns
    assert "nsib_support" in df.columns
    assert frac_siblings >= 0.5
    df["frac_siblings"] = df["nsib_support"] / (df["nsibs"] - 1)
    telomere_dict = (
        telomere_df.groupby("chrom").agg({"start": "min", "end": "max"}).to_dict()
    )
    telomere_dists = np.zeros(df.shape[0])
    for i, (chrom, pos) in tqdm(enumerate(zip(df.chrom.values, df.avg_pos.values))):
        telomere_dists[i] = telomere_dist(chrom, pos, telomere_dict=telomere_dict)
    df["telomere_dist"] = telomere_dists
    filt_df = (
        df[df["frac_siblings"] > frac_siblings]
        .groupby(["mother", "father", "child", "chrom", "crossover_sex"])
        .agg({"telomere_dist": "min"})
        .reset_index()
        .groupby(["mother", "father", "crossover_sex"])
        .agg({"telomere_dist": "mean"})
        .reset_index()
    )
    mother_df = filt_df[filt_df.crossover_sex == "maternal"][
        ["mother", "mother", "telomere_dist"]
    ]
    mother_df.columns = ["FID", "IID", "TelomereDist"]
    father_df = filt_df[filt_df.crossover_sex == "paternal"][
        ["father", "father", "telomere_dist"]
    ]
    father_df.columns = ["FID", "IID", "TelomereDist"]
    tot_df = pd.concat([mother_df, father_df])
    return tot_df


def avg_replication_timing(df, rt_df):
    """Compute phenotypes for replication timing."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert "chrom" in df.columns
    assert "avg_pos" in df.columns
    assert "nsibs" in df.columns
    assert "nsib_support" in df.columns
    # 1. Create the replication timing dictionary
    rt_agg_dict = rt_dict(rt_df)
    # 2. Iterate through the rows of the dataframe
    rt_vector = np.zeros(df.shape[0])
    for i, (chrom, pos) in tqdm(
        enumerate(zip(df["chrom"].values, df["avg_pos"].values))
    ):
        rt_vector[i] = rt_agg_dict[chrom](pos)
    df["ReplicationTiming"] = rt_vector
    mother_df = (
        df[df.crossover_sex == "maternal"]
        .groupby("mother")["ReplicationTiming"]
        .agg("mean")
        .reset_index()[["mother", "mother", "ReplicationTiming"]]
    )
    mother_df.columns = ["FID", "IID", "ReplicationTiming"]
    father_df = (
        df[df.crossover_sex == "paternal"]
        .groupby("father")["ReplicationTiming"]
        .agg("mean")
        .reset_index()[["father", "father", "ReplicationTiming"]]
    )
    father_df.columns = ["FID", "IID", "ReplicationTiming"]
    tot_df = pd.concat([mother_df, father_df])
    return tot_df


def avg_gc_content(df, bw_file, window=500):
    """Calculate the average GC content in a window around the average crossover position."""
    assert window > 0
    bw = pyBigWig.open(bw_file)
    assert bw.isBigWig()
    # Create a vector for the GC content ...
    gc_content = np.zeros(df.shape[0])
    for i, (chrom, pos) in tqdm(
        enumerate(zip(df["chrom"].values, df["avg_pos"].values))
    ):
        gc_content[i] = bw.stats(
            chrom, int(pos - window), int(pos + window), type="mean"
        )[0]
    df["GcContent"] = gc_content
    mother_df = (
        df[df.crossover_sex == "maternal"]
        .groupby("mother")["GcContent"]
        .agg("mean")
        .reset_index()[["mother", "mother", "GcContent"]]
    )
    mother_df.columns = ["FID", "IID", "GcContent"]
    father_df = (
        df[df.crossover_sex == "paternal"]
        .groupby("father")["GcContent"]
        .agg("mean")
        .reset_index()[["father", "father", "GcContent"]]
    )
    father_df.columns = ["FID", "IID", "GcContent"]
    tot_df = pd.concat([mother_df, father_df])
    return tot_df


if __name__ == "__main__":
    """Create several location-based phenotypes for analysis of recombination."""
    co_df = pd.read_csv(snakemake.input["co_data"], sep="\t")
    centromere_df = pd.read_csv(snakemake.input["centromeres"], header=None, sep="\t")
    centromere_df.columns = ["chrom", "start", "end", "feature"]
    telomere_df = pd.read_csv(snakemake.input["telomeres"], header=None, sep="\t")
    telomere_df.columns = ["chrom", "start", "end", "feature"]
    rt_df = pd.read_csv(snakemake.input["replication_timing"], header=None, sep="\t")
    rt_df.columns = ["chrom", "start", "end", "rt"]
    rt_df["midpt"] = (rt_df["start"] + rt_df["end"]) / 2
    centromere_pheno_df = avg_dist_centromere(co_df, centromere_df)
    telomere_pheno_df = avg_dist_telomere(co_df, telomere_df)
    rt_pheno_df = avg_replication_timing(co_df, rt_df)
    gc_pheno_df = avg_gc_content(
        co_df, snakemake.input["gc_content"], window=snakemake.params["gc_window"]
    )
    data_frames = [centromere_pheno_df, telomere_pheno_df, rt_pheno_df, gc_pheno_df]
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["FID", "IID"], how="outer"),
        data_frames,
    )
    if snakemake.params["plink_format"]:
        merged_df.rename(columns={"FID": "#FID"}, inplace=True)
    merged_df.to_csv(snakemake.output["pheno"], sep="\t", na_rep="NA", index=None)
