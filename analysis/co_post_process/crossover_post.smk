#!python3
import numpy as np
import pandas as pd
import polars as pl

import pickle, gzip
from tqdm import tqdm
from pathlib import Path
from io import StringIO


configfile: "config.yaml"


chroms = [f"chr{i}" for i in range(1, 23)]


rule all:
    input:
        expand(
            "results/{name}.crossover_filt.{recmap}.merged.meta.tsv.gz",
            name=config["crossover_data"].keys(),
            recmap=config["recomb_maps"].keys(),
        ),
        expand(
            "results/{name}.crossover_filt.{recmap}.crossover_count.{sex}.{euploid}.csv.gz",
            name=config["crossover_data"].keys(),
            recmap=config["recomb_maps"].keys(),
            sex=["maternal", "paternal"],
            euploid=["euploid", "aneuploid"],
        ),


# ---------------- Analysis 1a. Conduct preprocessing analyses. -------- #
rule filter_co_dataset:
    """Filter a crossover dataset according to the primary criteria."""
    input:
        crossover_data=lambda wildcards: config["crossover_data"][wildcards.name][
            "crossover_calls"
        ],
    output:
        co_filt_data="results/{name}.crossover_filt.tsv.gz",
        co_raw_data="results/{name}.crossover_raw.tsv.gz",
    params:
        qual_thresh=lambda wildcards: config["crossover_data"][wildcards.name][
            "qual_thresh"
        ],
    run:
        co_df = pl.read_csv(input.crossover_data, separator="\t")
        co_df = co_df.with_columns(
            pl.concat_str(
                [
                    pl.col("mother"),
                    pl.col("father"),
                    pl.col("child"),
                ],
                separator="+",
            ).alias("uid"),
        )
        co_df = co_df.with_columns(
            ((pl.col("min_pos_qual") + pl.col("max_pos_qual")) / 2).alias("qual_score"),
            (pl.col("nsib_support") / (pl.col("nsibs") - 1)).alias("frac_siblings"),
            pl.struct("uid", "chrom", "crossover_sex", "min_pos", "max_pos")
            .is_first_distinct()
            .alias("valid_co"),
        )
        valid_co_df = co_df.filter(
            pl.col("valid_co") & (pl.col("qual_score") > params.qual_thresh)
        )
        with gzip.open(output.co_filt_data, "wb") as f:
            valid_co_df.write_csv(f, separator="\t")
        with gzip.open(output.co_raw_data, "wb") as fp:
            co_df.write_csv(fp, separator="\t")


rule isolate_euploid_crossovers:
    """Isolate only the euploid chromosomes."""
    input:
        aneuploidy_tsv=config["aneuploidy_data"],
        co_filt_tsv="results/{name}.crossover_filt.tsv.gz",
    output:
        co_euploid_filt_tsv="results/{name}.crossover_filt.euploid_only.tsv.gz",
        co_aneuploid_filt_tsv="results/{name}.crossover_filt.aneuploid_only.tsv.gz",
        co_full_filt_tsv="results/{name}.crossover_filt.raw.tsv.gz",
    params:
        ppThresh=0.90,
    script:
        "scripts/isolate_euploids.py"


rule interpolate_co_locations:
    """Interpolate the locations of crossovers from crossover specific maps."""
    input:
        co_map="results/{name}.crossover_filt.{ploid}_only.tsv.gz",
        recmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
    output:
        co_map_interp="results/{name}.crossover_filt.{recmap}.{ploid}_only.tsv.gz",
    script:
        "scripts/interp_recmap.py"


rule intersect_w_metadata:
    """Intersect the crossover data with the resulting metadata."""
    input:
        co_map_interp_tsv="results/{name}.crossover_filt.{recmap}.{ploid}_only.tsv.gz",
        meta_csv=config["metadata"],
    wildcard_constraints:
        ploid="euploid|aneuploid",
    output:
        co_meta_map_tsv="results/{name}.crossover_filt.{recmap}.{ploid}_only.meta.tsv.gz",
    run:
        import pandas as pd

        meta_df = pd.read_csv(input.meta_csv)
        meta_df["egg_donor_bool"] = meta_df.egg_donor == "yes"
        meta_df["sperm_donor_bool"] = meta_df.sperm_donor == "yes"
        co_df = pd.read_csv(input.co_map_interp_tsv, sep="\t")
        # Only get the mother data ...
        meta_mother_df = (
            meta_df[meta_df.family_position == "mother"][
                [
                    "array",
                    "patient_age",
                    "partner_age",
                    "egg_donor_bool",
                    "sperm_donor_bool",
                    "year",
                ]
            ]
            .groupby("array")
            .agg("median")
            .reset_index()
        )
        meta_mother_df.rename(
            columns={
                "array": "mother",
                "egg_donor_bool": "egg_donor",
                "sperm_donor_bool": "sperm_donor",
            },
            inplace=True,
        )
        merged_meta_valid_co_df = co_df.merge(meta_mother_df, how="left")
        merged_meta_valid_co_df.to_csv(output.co_meta_map_tsv, index=None, sep="\t")


rule merge_euploid_aneuploid:
    input:
        euploid_tsv="results/{name}.crossover_filt.{recmap}.euploid_only.meta.tsv.gz",
        aneuploid_tsv="results/{name}.crossover_filt.{recmap}.aneuploid_only.meta.tsv.gz",
    output:
        merged_tsv="results/{name}.crossover_filt.{recmap}.merged.meta.tsv.gz",
    run:
        euploid_df = pd.read_csv(input.euploid_tsv, sep="\t")
        aneuploid_df = pd.read_csv(input.aneuploid_tsv, sep="\t")
        merged_df = pd.concat([euploid_df, aneuploid_df])
        merged_df.to_csv(output.merged_tsv, sep="\t", index=None)


rule estimate_crossover_counts:
    input:
        crossover_fp=rules.merge_euploid_aneuploid.output.merged_tsv,
        aneuploidy_tsv=config["aneuploidy_data"],
        genmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
        covariates=config["covariates"],
    output:
        maternal_co_count="results/{name}.crossover_filt.{recmap}.crossover_count.maternal.{euploid}.csv.gz",
        paternal_co_count="results/{name}.crossover_filt.{recmap}.crossover_count.paternal.{euploid}.csv.gz",
    wildcard_constraints:
        euploid="euploid|aneuploid",
    params:
        euploid=lambda wildcards: wildcards.euploid == "euploid",
    script:
        "scripts/gen_co_counts.py"


rule estimate_centromere_telomere_dist:
    input:
        crossover_fp=rules.merge_euploid_aneuploid.output.merged_tsv,
        aneuploidy_tsv=config["aneuploidy_data"],
        genmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
        covariates=config["covariates"],
    output:
        maternal_co_dist="results/{name}.crossover_filt.{recmap}.centromere_telomere_dist.maternal.{euploid}.csv.gz",
        paternal_co_dist="results/{name}.crossover_filt.{recmap}.centromere_telomere_dist.paternal.{euploid}.csv.gz",
    wildcard_constraints:
        euploid="euploid|aneuploid",
    params:
        euploid=lambda wildcards: wildcards.euploid == "euploid",
    script:
        "scripts/gen_rec_location.py"


rule create_sex_specific_hotspots:
    """Create sex-specific hotspot files from Haldorsson et al 2019."""
    input:
        genmap=lambda wildcards: config["hotspots"][wildcards.sex],
    output:
        hotspots="results/hotspots/{name}.{sex}.hotspots.tsv",
    wildcard_constraints:
        sex="Male|Female",
    params:
        srr=10,
    resources:
        time="1:00:00",
        mem_mb="8G",
    run:
        df = pd.read_csv(input.genmap, sep="\t", comment="#")
        df["SRR"] = df.cMperMb / df.cMperMb.mean()
        filt_df = df[df.SRR > params.srr]
        filt_df.rename(
            columns={"Chr": "chrom", "Begin": "start", "End": "end"}, inplace=True
        )
        filt_df.to_csv(output.hotspots, index=None, sep="\t")


rule estimate_hotspot_occupancy:
    input:
        crossover_fp=rules.merge_euploid_aneuploid.output.merged_tsv,
        male_hotspots="results/hotspots/{name}.Male.hotspots.tsv",
        female_hotspots="results/hotspots/{name}.Female.hotspots.tsv",
        aneuploidy_tsv=config["aneuploidy_data"],
        genmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
        covariates=config["covariates"],
    output:
        maternal_occupancy="results/{name}.crossover_filt.{recmap}.hotspot_occupy.maternal.{euploid}.csv.gz",
        paternal_occupancy="results/{name}.crossover_filt.{recmap}.hotspot_occupy.paternal.{euploid}.csv.gz",
    wildcard_constraints:
        euploid="euploid|aneuploid",
    params:
        euploid=lambda wildcards: wildcards.euploid == "euploid",
        max_interval=50e3,
        nreps=100,
        ngridpts=300,
    script:
        "scripts/gen_hotspot_occupancy.py"
