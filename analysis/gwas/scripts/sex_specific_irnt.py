import numpy as np
import pandas as pd
from utils import inverse_rank_transform

if __name__ == "__main__":
    # 1. Read in the data frames
    covar_df = pd.read_csv(snakemake.input["covar"], sep="\t", na_values=["NA"])
    assert "Sex" in covar_df.columns
    pheno_raw_df = pd.read_csv(snakemake.input["pheno"], sep="\t", na_values=["NA"])
    # 2. Isolate the male and female samples ...
    female_ids = covar_df[covar_df.Sex == 0]
    male_ids = covar_df[covar_df.Sex == 1]
    if snakemake.wildcards["format"] == "plink2":
        female_raw_pheno_df = pheno_raw_df[pheno_raw_df.IID.isin(female_ids.IID)]
        male_raw_pheno_df = pheno_raw_df[pheno_raw_df.IID.isin(male_ids.IID)]
        for c in female_raw_pheno_df.columns:
            if c not in ["#FID", "IID"]:
                female_raw_pheno_df[c] = inverse_rank_transform(
                    female_raw_pheno_df[c].values
                )
        for c in male_raw_pheno_df.columns:
            if c not in ["#FID", "IID"]:
                male_raw_pheno_df[c] = inverse_rank_transform(
                    male_raw_pheno_df[c].values
                )
        pheno_irnt_df = pd.concat([female_raw_pheno_df, male_raw_pheno_df])
    else:
        female_raw_pheno_df = pheno_raw_df[pheno_raw_df.IID.isin(female_ids.IID)]
        male_raw_pheno_df = pheno_raw_df[pheno_raw_df.IID.isin(male_ids.IID)]
        for c in female_raw_pheno_df.columns:
            if c not in ["FID", "IID"]:
                female_raw_pheno_df[c] = inverse_rank_transform(
                    female_raw_pheno_df[c].values
                )
        for c in male_raw_pheno_df.columns:
            if c not in ["FID", "IID"]:
                male_raw_pheno_df[c] = inverse_rank_transform(
                    male_raw_pheno_df[c].values
                )
        pheno_irnt_df = pd.concat([female_raw_pheno_df, male_raw_pheno_df])
    # 3. Write out the file
    pheno_irnt_df.to_csv(snakemake.output["pheno"], sep="\t", index=None, na_rep="NA")
