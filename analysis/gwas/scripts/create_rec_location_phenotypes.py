#!python3

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def centromere_dist(chrom, pos, centromere_dict):
    pts = np.array([centromere_dict["start"][chrom], centromere_dict["end"][chrom]])
    dist = np.min(np.abs(pos - pts))
    return dist


def telomere_dist(chrom, pos, telomere_dict):
    pts = np.array([telomere_dict["start"][chrom], telomere_dict["end"][chrom]])
    dist = np.min(np.abs(pos - pts))
    return dist


def avg_dist_centromere(df, centromere_df):
    """Compute the average distance in bp from centromere of the nearest crossover."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert "chrom" in df.columns
    assert "avg_pos" in df.columns
    centromere_dict = (
        centromere_df.groupby("chrom").agg({"start": np.min, "end": np.max}).to_dict()
    )
    centromere_dists = np.zeros(df.shape[0])
    for i, (chrom, pos) in tqdm(enumerate(zip(df.chrom.values, df.avg_pos.values))):
        centromere_dists[i] = centromere_dist(
            chrom, pos, centromere_dict=centromere_dict
        )
    df["centromere_dist"] = centromere_dists
    filt_df = (
        df.groupby(["mother", "father", "child", "chrom", "crossover_sex"])
        .agg({"centromere_dist": np.min})
        .reset_index()
        .groupby(["mother", "father", "crossover_sex"])
        .agg({"centromere_dist": np.mean})
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


def avg_dist_telomere(df, telomere_df):
    """Compute the average distance in bp from the telomeres of the nearest crossover."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert "chrom" in df.columns
    assert "avg_pos" in df.columns
    telomere_dict = (
        telomere_df.groupby("chrom").agg({"start": np.min, "end": np.max}).to_dict()
    )
    telomere_dists = np.zeros(df.shape[0])
    for i, (chrom, pos) in tqdm(enumerate(zip(df.chrom.values, df.avg_pos.values))):
        telomere_dists[i] = telomere_dist(chrom, pos, telomere_dict=telomere_dict)
    df["telomere_dist"] = telomere_dists
    filt_df = (
        df.groupby(["mother", "father", "child", "chrom", "crossover_sex"])
        .agg({"telomere_dist": np.min})
        .reset_index()
        .groupby(["mother", "father", "crossover_sex"])
        .agg({"telomere_dist": np.mean})
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


if __name__ == "__main__":
    """Create several location-based phenotypes for analysis of recombination."""
    co_df = pd.read_csv(snakemake.input["co_data"], sep="\t")
    centromere_df = pd.read_csv(snakemake.input["centromeres"], header=None, sep="\t")
    centromere_df.columns = ["chrom", "start", "end", "feature"]
    telomere_df = pd.read_csv(snakemake.input["telomeres"], header=None, sep="\t")
    telomere_df.columns = ["chrom", "start", "end", "feature"]
    centromere_pheno_df = avg_dist_centromere(co_df, centromere_df)
    telomere_pheno_df = avg_dist_telomere(co_df, telomere_df)
    merged_df = centromere_pheno_df.merge(telomere_pheno_df, on=["FID", "IID"])
    if snakemake.params["plink_format"]:
        merged_df.rename(columns={"FID": "#FID"}, inplace=True)
    merged_df.to_csv(snakemake.output["pheno"], sep="\t", index=None)
