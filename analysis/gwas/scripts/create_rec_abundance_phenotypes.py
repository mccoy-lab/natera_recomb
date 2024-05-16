#!python3

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

cv = lambda x: np.nanstd(x) / np.nanmean(x)


def mean_var_co_per_genome(df, randomize=True, seed=42):
    """Compute the average number of crossovers per-chromosome for an individual."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert "chrom" in df.columns
    co_df = (
        df.groupby(["mother", "father", "child", "crossover_sex"])
        .count()
        .reset_index()
        .rename(columns={"chrom": "n_crossover"})[
            ["mother", "father", "child", "crossover_sex", "n_crossover"]
        ]
    )
    mat_data = []
    for m in tqdm(np.unique(co_df.mother)):
        values = co_df[(co_df.mother == m) & (co_df.crossover_sex == "maternal")][
            "n_crossover"
        ].values
        mat_data.append([m, m, np.nanmean(values), np.nanvar(values), cv(values)])
    mat_df = pd.DataFrame(mat_data)
    mat_df.columns = ["FID", "IID", "MeanCO", "VarCO", "cvCO"]
    pat_data = []
    for p in tqdm(np.unique(df.father)):
        values = co_df[(co_df.father == p) & (co_df.crossover_sex == "paternal")][
            "n_crossover"
        ].values
        pat_data.append([p, p, np.nanmean(values), np.nanvar(values), cv(values)])
    pat_df = pd.DataFrame(pat_data)
    pat_df.columns = ["FID", "IID", "MeanCO", "VarCO", "cvCO"]
    if randomize:
        np.random.seed(seed)
        pat_df["RandMeanCO"] = np.random.permutation(pat_df["MeanCO"].values)
        mat_df["RandMeanCO"] = np.random.permutation(mat_df["MeanCO"].values)
    out_df = pd.concat([mat_df, pat_df])
    return out_df


def random_pheno(df, seed=42):
    """Create a random phenotype as a test for main analysis."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert seed > 0
    np.random.seed(seed)
    data = []
    for m in np.unique(df.mother):
        x = np.random.normal()
        data.append([m, m, x])
    for p in np.unique(df.father):
        x = np.random.normal()
        data.append([p, p, x])
    out_df = pd.DataFrame(data)
    out_df.columns = ["FID", "IID", "RandPheno"]
    return out_df


if __name__ == "__main__":
    """Create several phenotypes for analysis of recombination."""
    co_df = pd.read_csv(snakemake.input["co_data"], sep="\t")
    mean_R_df = mean_var_co_per_genome(co_df)
    rand_df = random_pheno(co_df)
    merged_df = mean_R_df.merge(rand_df)
    if snakemake.params["plink_format"]:
        merged_df.rename(columns={"FID": "#FID"}, inplace=True)
    merged_df.to_csv(snakemake.output["pheno"], sep="\t", index=None)
