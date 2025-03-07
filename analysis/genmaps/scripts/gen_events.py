import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    co_df = pd.read_csv(snakemake.input["co_map"], sep="\t")
    sex_spec_co_df = co_df[co_df.crossover_sex == snakemake.params["sex"]]

    # Estimate the events files per-chromosome ...
    sex_spec_co_chrom_df = (
        sex_spec_co_df[sex_spec_co_df.chrom == snakemake.wildcards["chrom"]][
            ["min_pos", "max_pos"]
        ]
        .dropna()
        .astype(int)
    )
    # fname = f"results/{snakemake.wildcards['sex']}_genmap/{snakemake.wildcards['name']}.events.{snakemake.wildcards['chrom']}.{snakemake.wildcards['sex']}.txt"
    sex_spec_co_chrom_df.to_csv(
        snakemake.output["events"], index=None, header=None, sep="\t"
    )
