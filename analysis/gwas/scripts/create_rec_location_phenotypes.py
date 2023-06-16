#!python3

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def avg_dist_centromere(df, centromere_df):
    """Compute the average distance in bp from centromere of the nearest crossover."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert "chrom" in df.columns
    pass

def avg_dist_telomere(df, telomere_df):
    """Compute the average distance in bp from the telomeres of the nearest crossover."""
    assert "mother" in df.columns
    assert "father" in df.columns
    assert "child" in df.columns
    assert "crossover_sex" in df.columns
    assert "chrom" in df.columns
    pass

if __name__ == "__main__":
    """Create several phenotypes for analysis of recombination."""
    co_df = pd.read_csv(snakemake.input["co_data"], sep="\t")
    mean_R_df = mean_var_co_per_genome(co_df)
    rand_df = random_pheno(co_df)
    merged_df = mean_R_df.merge(rand_df)
    if snakemake.params["plink_format"]:
        merged_df.rename(columns={"FID": "#FID"})
    merged_df.to_csv(snakemake.output["pheno"], sep="\t", index=None)
