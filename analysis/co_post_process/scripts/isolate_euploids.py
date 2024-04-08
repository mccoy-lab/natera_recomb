import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Keep track at the chromosome -level the number of disomies able to be processed ...
    aneuploidy_df = pd.read_csv(snakemake.input["aneuploidy_tsv"], sep="\t")
    aneuploidy_df["uid"] = (
        aneuploidy_df["mother"] + aneuploidy_df["father"] + aneuploidy_df["child"]
    )
    aneuploidy_df["parents"] = aneuploidy_df["mother"] + aneuploidy_df["father"]
    aneuploidy_df["disomy"] = aneuploidy_df["2"] >= snakemake.params["ppThresh"]

    # Add in the putative euploid embryos (must have all 22 chromosomes as strongly euploid ....
    euploid_df = (
        aneuploidy_df[aneuploidy_df.disomy]
        .groupby(["mother", "father", "child"])["chrom"]
        .agg(lambda x: x.size)
        .reset_index()
        .rename(columns={"chrom": "n_euploid_chrom"})
    )
    euploid_df["euploid"] = euploid_df.n_euploid_chrom == 22
    # Filter to the appropriate set of UIDs
    aneuploidy_df = aneuploidy_df.merge(euploid_df, how="left")
    aneuploidy_df = aneuploidy_df[~aneuploidy_df.embryo_noise_3sd]
    # Load in the crossover dataframe and isolate the euploid embryos
    co_df = pd.read_csv(snakemake.input["co_filt_tsv"], sep="\t")
    euploid_embryo_id = aneuploidy_df[aneuploidy_df.euploid == True]["uid"].unique()
    # Write out the euploid embryo locations ...
    co_df[co_df.uid.isin(euploid_embryo_id)].to_csv(
        snakemake.output["co_euploid_filt_tsv"], index=False, sep="\t"
    )
