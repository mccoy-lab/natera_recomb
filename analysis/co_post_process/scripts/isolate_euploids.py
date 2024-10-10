import numpy as np
import polars as pl
import gzip


def obtain_euploid_aneuploid(aneuploidy_fp, ppThresh=0.90, threads=8):
    karyohmm_dtypes = {
        "sigma_baf": pl.Float32,
        "pi0_baf": pl.Float32,
        "0": pl.Float32,
        "1m": pl.Float32,
        "1p": pl.Float32,
        "2": pl.Float32,
        "3m": pl.Float32,
        "3p": pl.Float32,
        "bf_max": pl.Float32,
    }

    aneuploidy_df = pl.read_csv(
        aneuploidy_fp,
        infer_schema_length=10000,
        dtypes=karyohmm_dtypes,
        n_threads=threads,
        null_values=["NA"],
        separator="\t",
    )

    aneuploidy_df = aneuploidy_df.with_columns(
        pl.concat_str(
            [
                pl.col("mother"),
                pl.col("father"),
                pl.col("child"),
            ],
            separator="+",
        ).alias("uid"),
    )
    # Filter out the Day 3 embryos and ones that have nan
    aneuploidy_df = aneuploidy_df.filter(
        ~pl.col("day3_embryo") | (pl.col("1m").is_nan())
    )
    # Now get the euploid uids
    n_aneuploid_chrom_df = (
        aneuploidy_df.group_by("uid")
        .agg(
            (
                (pl.col("bf_max") > 5)
                & (pl.col("post_max") > ppThresh)
                & (pl.col("bf_max_cat") != "2")
            ).sum()
        )
        .with_columns(pl.col("bf_max").alias("n_aneuploid_chrom"))[
            ["uid", "n_aneuploid_chrom"]
        ]
    )
    n_euploid_chrom_df = (
        aneuploidy_df.group_by("uid")
        .agg(
            (
                (pl.col("bf_max") > 5)
                & (pl.col("post_max") > ppThresh)
                & (pl.col("bf_max_cat") == "2")
            ).sum()
        )
        .with_columns(pl.col("bf_max").alias("n_euploid_chrom"))[
            ["uid", "n_euploid_chrom"]
        ]
    )
    euploid_uids = (
        n_euploid_chrom_df.filter(pl.col("n_euploid_chrom") == 22)["uid"]
        .unique()
        .to_numpy()
    )
    aneuploid_uids = (
        n_aneuploid_chrom_df.filter(pl.col("n_aneuploid_chrom") > 0)["uid"]
        .unique()
        .to_numpy()
    )
    return euploid_uids, aneuploid_uids


if __name__ == "__main__":
    # Keep track at the chromosome -level the number of disomies able to be processed ...
    euploid_uids, aneuploid_uids = obtain_euploid_aneuploid(
        snakemake.input["aneuploidy_tsv"],
        ppThresh=snakemake.params["ppThresh"],
        threads=snakemake.threads,
    )
    # Load in the crossover dataframe and isolate the euploid embryos
    co_df = pl.read_csv(snakemake.input["co_filt_tsv"], separator="\t")
    co_df = co_df.with_columns(
        pl.concat_str(
            [
                pl.col("mother"),
                pl.col("father"),
                pl.col("child"),
            ],
            separator="+",
        ).alias("uid")
    )
    # Assign euploid aneuploid status ...
    co_df = co_df.with_columns(
        pl.col("uid").is_in(euploid_uids).alias("euploid"),
        pl.col("uid").is_in(aneuploid_uids).alias("aneuploid"),
    )
    # Write out the euploid embryo locations ...
    with gzip.open(snakemake.output["co_euploid_filt_tsv"], "wb") as f_eu:
        co_df.filter(pl.col("euploid")).write_csv(f_eu, separator="\t")
    # Write out the aneuploid embryo locations ...
    with gzip.open(snakemake.output["co_aneuploid_filt_tsv"], "wb") as f_aneu:
        co_df.filter(pl.col("aneuploid")).write_csv(f_aneu, separator="\t")
    # Write out the full crossover call data
    with gzip.open(snakemake.output["co_full_filt_tsv"], "wb") as f_tot:
        co_df.write_csv(f_tot, separator="\t")
