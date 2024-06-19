from pathlib import Path

import numpy as np
import pandas as pd
import re

if __name__ == "__main__":
    x = Path(snakemake.input["sumstats"])
    spltname = re.split("\_|\.", x.name)
    sex = spltname[3]
    pheno = spltname[5]
    df = pd.read_csv(snakemake.input["sumstats"], header=None, sep="\t")
    df.columns = [
        "CHROM",
        "POS",
        "ID",
        "P",
        "TOTAL",
        "NONSIG",
        "S0.05",
        "S0.01",
        "S0.001",
        "S0.0001",
        "SP2",
        "POS_A",
        "POS_B",
        "CHROM_X",
        "GeneStart",
        "GeneEnd",
        "Gencode",
        "Dist",
    ]
    df["PHENO"] = f"{pheno}_{sex}"
    freq_df = pd.read_csv(snakemake.input["freqs"], sep="\t")
    freq_df.rename(columns={"#CHROM": "CHROM"}, inplace=True)
    beta_df = pd.read_csv(snakemake.input["top_variants"], sep="\t")
    beta_df = beta_df.merge(
        freq_df[["ID", "REF", "ALT", "ALT_FREQS"]],
        on=["ID", "REF", "ALT"],
        how="left",
    )
    df = df.merge(
        beta_df[
            [
                "ID",
                "REF",
                "ALT",
                "A1",
                "BETA",
                "SE",
                "T_STAT",
                "OBS_CT",
                "ALT_FREQS",
            ]
        ],
        on=["ID"],
        how="left",
    )
    final_df = df[
        [
            "PHENO",
            "ID",
            "P",
            "REF",
            "ALT",
            "ALT_FREQS",
            "OBS_CT",
            "BETA",
            "SE",
            "A1",
            "T_STAT",
            "TOTAL",
            "NONSIG",
            "S0.05",
            "S0.01",
            "S0.001",
            "S0.0001",
            "SP2",
            "GeneStart",
            "GeneEnd",
            "Gencode",
            "Dist",
        ]
    ]
    final_df.to_csv(snakemake.output["final_sumstats"], sep="\t", index=None)
