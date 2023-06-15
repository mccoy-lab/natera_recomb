import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    co_df = pd.read_csv(snakemake.input["co_map_interp"], sep="\t")
    sex_spec_co_df = co_df[co_df.crossover_sex == snakemake.params["sex"]]
    recmap_df = pd.read_csv(snakemake.input["recmap"], comment="#", sep="\t")
    recmap_df.columns = ["chrom", "begin", "end", "cMperMb", "cM"]
    nmeioses = np.unique(sex_spec_co_df.child.values).size

    # Estimate the events files per-chromosome ...
    for c in tqdm(np.unique(sex_spec_co_df.chrom.values)):
        region_chrom_df = recmap_df[recmap_df.chrom == c][["begin", "end"]]
        region_chrom_df["nbmeioses"] = nmeioses
        fname = f"results/{snakemake.wildcards['sex']}_genmap/{snakemake.wildcards['name']}.nbmeioses.{snakemake.wildcards['recmap']}.{c}.{snakemake.wildcards['sex']}.txt"
        region_chrom_df.to_csv(fname, index=None, header=None, sep="\t")
