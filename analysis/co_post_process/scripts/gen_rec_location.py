#!python3

from functools import reduce

import numpy as np
import polars as pl
import pyBigWig
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm


def centromere_dist(chrom, pos, centromere_dict, size_dict):
    pts = np.array([centromere_dict["start"][chrom], centromere_dict["end"][chrom]])
    dist = np.min(np.abs(pos - pts)) / size_dict[chrom]
    return dist


def telomere_dist(chrom, pos, telomere_dict, size_dict):
    pts = np.array([telomere_dict["start"][chrom], telomere_dict["end"][chrom]])
    dist = np.min(np.abs(pos - pts)) / size_dict[chrom]
    return dist


if __name__ == "__main__":
    """Create several location-based phenotypes for analysis of recombination."""
    # Read in the sanitized and filtered crossover dataframe
    aneuploidy_df = pl.read_csv(
        snakemake.input["aneuploidy_tsv"], separator="\t", null_values=["NA"]
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

    # Read in the centromere + telomere dataframes
    centromere_df = pl.read_csv(
        snakemake.input["centromeres"], separator="\t", has_header=False
    )
    centromere_df.columns = ["chrom", "start", "end", "feature"]
    telomere_df = pl.read_csv(
        snakemake.input["telomeres"], separator="\t", has_header=False
    )
    telomere_df.columns = ["chrom", "start", "end", "feature"]
    size_df = pl.read_csv(
        snakemake.input["chromsize"], separator="\t", has_header=False
    )
    size_df.columns = ["chrom", "size"]
    # Create the centromere and telomere lookup-dictionaries
    telomere_dict = {"start": {}, "end": {}}
    for chrom, start, end in (
        telomere_df.group_by("chrom")
        .agg(pl.col("start").min(), pl.col("end").max())
        .rows()
    ):
        telomere_dict["start"][chrom] = start
        telomere_dict["end"][chrom] = end

    centromere_dict = {"start": {}, "end": {}}
    for chrom, start, end in (
        centromere_df.group_by("chrom")
        .agg(pl.col("start").min(), pl.col("end").max())
        .rows()
    ):
        centromere_dict["start"][chrom] = start
        centromere_dict["end"][chrom] = end

    size_dict = {}
    for chrom, size in size_df.rows():
        size_dict[chrom] = size
    # Estimate the per-crossover centromere / telomere distance, scaled by chromosome size
    crossover_df = crossover_df.with_columns(
        pl.struct(["chrom", "avg_pos"])
        .map_elements(
            lambda x: centromere_dist(
                x["chrom"],
                x["avg_pos"],
                centromere_dict=centromere_dict,
                size_dict=size_dict,
            ),
            return_dtype=pl.Float32,
        )
        .alias("centromere_dist"),
        pl.struct(["chrom", "avg_pos"])
        .map_elements(
            lambda x: telomere_dist(
                x["chrom"],
                x["avg_pos"],
                telomere_dict=telomere_dict,
                size_dict=size_dict,
            ),
            return_dtype=pl.Float32,
        )
        .alias("telomere_dist"),
        pl.struct(["chrom"])
        .map_elements(lambda x: size_dict[x["chrom"]], return_dtype=pl.Float32)
        .alias("chrom_size"),
    )
    if snakemake.params["euploid"]:
        maternal_euploid_co_df = (
            crossover_df.filter(
                pl.col("euploid") & (pl.col("crossover_sex") == "maternal")
            )
            .group_by("chrom", "uid", "mother")
            .agg(
                pl.col("avg_pos"),
                pl.col("centromere_dist").mean(),
                pl.col("telomere_dist").mean(),
                pl.col("chrom_size").mean(),
                pl.col("avg_pos").count().alias("nco"),
                pl.col("patient_age").mean(),
                pl.col("egg_donor").first(),
                pl.col("sperm_donor").first(),
            )
        )
    else:
        pass
