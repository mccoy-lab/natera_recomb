#!python3
import numpy as np
import pandas as pd

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


# expand(
#     "results/xo_interference/{name}.age_xo_interference.{recmap}.{chrom}.tsv",
#     name=config["crossover_data"].keys(),
#     recmap=config["recomb_maps"].keys(),
#     chrom=["chr4", "chr10", "chr22"],
# ),
# expand(
#     "results/{sex}_genmap/{name}.{chrom}.{sex}.{raw}-rates.txt",
#     sex=["maternal", "paternal"],
#     raw=["raw", "split"],
#     name=config["crossover_data"].keys(),
#     chrom=["chr4", "chr10", "chr22"],
# ),


# ---------------- Analysis 1a. Conduct preprocessing analyses. -------- #
rule filter_co_dataset:
    """Filter a crossover dataset according to the primary criteria."""
    input:
        crossover_data=lambda wildcards: config["crossover_data"][wildcards.name],
    output:
        co_filt_data="results/{name}.crossover_filt.tsv.gz",
    run:
        import pandas as pd

        co_df = pd.read_csv(input.crossover_data, sep="\t")
        co_df["uid"] = co_df["mother"] + co_df["father"] + co_df["child"]
        co_df["valid_co"] = ~co_df.duplicated(
            ["uid", "chrom", "crossover_sex", "min_pos", "max_pos"], keep=False
        )
        valid_co_df = co_df[co_df["valid_co"]]
        valid_co_df.to_csv(output.co_filt_data, sep="\t", index=None)


rule isolate_euploid_crossovers:
    """Isolate only the euploid chromosomes."""
    input:
        aneuploidy_tsv=config["aneuploidy_data"],
        co_filt_tsv="results/{name}.crossover_filt.tsv.gz",
    output:
        co_euploid_filt_tsv="results/{name}.crossover_filt.euploid_only.tsv.gz",
        co_aneuploid_filt_tsv="results/{name}.crossover_filt.aneuploid_only.tsv.gz",
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
        euploid_df["aneuploid"] = False
        aneuploid_df["aneuploid"] = True
        merged_df = pd.concat([euploid_df, aneuploid_df])
        merged_df.to_csv(output.merged_tsv, sep="\t", index=None)

rule estimate_chrom_specific_aneuploidy_effect:
    """Estimate the effects of multiple linear models."""
    input:
        merged_tsv="results/{name}.crossover_filt.{recmap}.merged.meta.tsv.gz",
    output:
        mean_per_chrom_effects = "results/statistical_models/aneuploidy_effect_per_chrom.mean.tsv",
        var_per_chrom_effects = "results/statistical_models/aneuploidy_effect_per_chrom.var.tsv"
    shell:
        """
        Rscript scripts/aneuploidy_effect_per_chrom.R {input.merged_tsv}
        """


# ------- Analysis 2. Estimate Crossover Interference Stratified by Age & Sex -------- #
rule age_sex_stratified_co_interference:
    input:
        co_meta_map_tsv="results/{name}.crossover_filt.{recmap}.{ploid}_only.meta.tsv.gz",
        recmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
    output:
        age_sex_interference="results/xo_interference/{name}.age_xo_interference.{recmap}.{chrom}.tsv",
    params:
        nbins=10,
        nboots=5,
        seed=42,
    script:
        "scripts/est_age_strat_xo.py"


# ------- Analysis 3. Posterior estimates of CO-interference across individuals. ------- #


# ------- Analysis 4. Estimation of sex-specific recombination maps from crossover data ------ #
rule split_sex_specific_co_data:
    """Splits crossovers into maternal/paternal events."""
    input:
        co_map="results/{name}.crossover_filt.tsv.gz",
    output:
        "results/{sex}_genmap/{name}.events.{chrom}.{sex}.txt",
    wildcard_constraints:
        sex="maternal|paternal",
    params:
        sex=lambda wildcards: wildcards.sex,
    script:
        "scripts/gen_events.py"


rule setup_intervals_co_data:
    """Setup intervals on which to estimate recombination rates."""
    input:
        co_map="results/{name}.crossover_filt.tsv.gz",
    output:
        "results/{sex}_genmap/{name}.nbmeioses.{chrom}.{sex}.{raw}.txt",
    wildcard_constraints:
        sex="maternal|paternal",
        raw="raw|split",
    params:
        sex=lambda wildcards: wildcards.sex,
        nsplit=5,
        use_raw=lambda wildcards: wildcards.raw == "raw",
    script:
        "scripts/gen_nbmeioses.py"


rule est_recomb_rate_rmcmc:
    input:
        events_file="results/{sex}_genmap/{name}.events.{chrom}.{sex}.txt",
        nbmeioses_file="results/{sex}_genmap/{name}.nbmeioses.{chrom}.{sex}.{raw}.txt",
        rMCMC="./rMCMC/rMCMC/rMCMC",
    output:
        rates_out="results/{sex}_genmap/{name}.{chrom}.{sex}.{raw}-rates.txt",
        events_out="results/{sex}_genmap/{name}.{chrom}.{sex}.{raw}-events.txt",
    params:
        outfix=lambda wildcards: f"results/{wildcards.sex}_genmap/{wildcards.name}.{wildcards.chrom}.{wildcards.sex}.{wildcards.raw}",
        nmeioses=lambda wildcards: pd.read_csv(
            f"results/{wildcards.sex}_genmap/{wildcards.name}.nbmeioses.{wildcards.chrom}.{wildcards.sex}.{wildcards.raw}.txt",
            nrows=1,
            sep="\s",
        ).values[:, 2][0],
    shell:
        "{input.rMCMC} -i {input.events_file} -nbmeioses {input.nbmeioses_file} -m {params.nmeioses} -o {params.outfix}"
