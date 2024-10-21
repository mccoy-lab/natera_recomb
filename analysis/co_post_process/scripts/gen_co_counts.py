import numpy as np
from scipy.stats import *
import polars as pl
import gzip as gz
from tqdm import tqdm

if __name__ == "__main__":
    # Read in the sanitized and filtered crossover dataframe
    crossover_df = pl.read_csv(snakemake.input["crossover_fp"], separator="\t")
    crossover_df = crossover_df.with_columns(
        pl.concat_str(
            [
                pl.col("mother"),
                pl.col("father"),
                pl.col("child"),
            ],
            separator="+",
        ).alias("uid"),
    )
    genmap_df = pl.read_csv(
        snakemake.input["genmap"], comment_prefix="#", separator="\t"
    )
    genlen_df = (
        genmap_df.group_by(pl.col("Chr"))
        .agg(pl.col("cM").max(), pl.col("End").max(), pl.col("Begin").min())
        .with_columns((pl.col("End") - pl.col("Begin")).alias("seq_len"))
        .rename({"Chr": "chrom", "cM": "cM_len"})
        .sort("seq_len")
    )
    covar_df = pl.read_csv(
        snakemake.input["covariates"], null_values="NA", separator="\t"
    )

    maternal_euploid_co_df = (
        crossover_df.filter(pl.col("euploid") & (pl.col("crossover_sex") == "maternal"))
        .group_by("chrom", "uid", "mother")
        .agg(
            pl.col("avg_pos").count(),
            pl.col("patient_age").mean(),
            pl.col("egg_donor").first(),
            pl.col("sperm_donor").first(),
            pl.col("avg_pi0").first(),
            pl.col("avg_sigma").first()
        )
    )

    chroms = [f"chr{i}" for i in range(1, 23)]
    uids = maternal_euploid_co_df["uid"].to_numpy()
    full_uid_df = pl.DataFrame(
        {
            "uid": uids,
            "chrom": [chroms for _ in uids],
            "avg_pos": [np.zeros(22, dtype=np.uint32) for _ in tqdm(uids)],
        }
    )
    full_uid_df = full_uid_df.explode("chrom").explode("avg_pos")
    uid_anti_join_df = full_uid_df.join(
        maternal_euploid_co_df, how="anti", on=["uid", "chrom"]
    ).unique()

    maternal_euploid_filled_co_df = pl.concat(
        [maternal_euploid_co_df, uid_anti_join_df], how="diagonal"
    )
    maternal_euploid_filled_co_df = (
        maternal_euploid_filled_co_df.sort("uid")
        .with_columns(
            pl.col("mother").forward_fill(),
            pl.col("patient_age").forward_fill(),
            pl.col("egg_donor").forward_fill(),
            pl.col("sperm_donor").forward_fill(),
        )
        .rename({"avg_pos": "nco", "mother": "IID"})
    )
    maternal_euploid_filled_co_merged_df = maternal_euploid_filled_co_df.join(
        genlen_df, on=["chrom"]
    ).join(covar_df, on=["IID"])
    with gz.open(snakemake.output["maternal_euploid"], "w+") as mat_out:
        maternal_euploid_filled_co_merged_df.write_csv(mat_out, null_value="NA")

    paternal_euploid_co_df = (
        crossover_df.filter(pl.col("euploid") & (pl.col("crossover_sex") == "paternal"))
        .group_by("chrom", "uid", "father")
        .agg(
            pl.col("avg_pos").count(),
            pl.col("partner_age").mean(),
            pl.col("egg_donor").first(),
            pl.col("sperm_donor").first(),
            pl.col("avg_pi0").first(),
            pl.col("avg_sigma").first()
        )
    )

    uids = paternal_euploid_co_df["uid"].to_numpy()
    full_uid_df = pl.DataFrame(
        {
            "uid": uids,
            "chrom": [chroms for _ in uids],
            "avg_pos": [np.zeros(22, dtype=np.uint32) for _ in uids],
        }
    )
    full_uid_df = full_uid_df.explode("chrom").explode("avg_pos")
    uid_anti_join_df = full_uid_df.join(
        paternal_euploid_co_df, how="anti", on=["uid", "chrom"]
    ).unique()

    paternal_euploid_filled_co_df = pl.concat(
        [paternal_euploid_co_df, uid_anti_join_df], how="diagonal"
    )
    paternal_euploid_filled_co_df = (
        paternal_euploid_filled_co_df.sort("uid")
        .with_columns(
            pl.col("father").forward_fill(),
            pl.col("partner_age").forward_fill(),
            pl.col("egg_donor").forward_fill(),
            pl.col("sperm_donor").forward_fill(),
        )
        .rename({"avg_pos": "nco", "father": "IID"})
    )

    paternal_euploid_filled_co_merged_df = paternal_euploid_filled_co_df.join(
        genlen_df, on=["chrom"]
    ).join(covar_df, on=["IID"])
    with gz.open(snakemake.output["paternal_euploid"], "w+") as pat_out:
        paternal_euploid_filled_co_merged_df.write_csv(pat_out, null_value="NA")
