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
        # expand(
        #     "results/{name}.age_xo_interference.{recmap}.tsv",
        #     name=config["crossover_data"].keys(),
        #     recmap=config["recomb_maps"].keys(),
        # ),
        expand(
            "results/{sex}_genmap/{name}.{recmap}.{chrom}.{sex}-rates.txt",
            sex="maternal",
            name=config["crossover_data"].keys(),
            recmap=config["recomb_maps"].keys(),
            chrom="chr22",
        ),


rule filter_co_dataset:
    """Filter a crossover dataset according to our key criteria."""
    input:
        crossover_data=lambda wildcards: config["crossover_data"][wildcards.name],
    output:
        co_filt_data="results/{name}.crossover_filt.tsv.gz",
    run:
        import pandas as pd

        co_df = pd.read_csv(input.crossover_data, sep="\t")
        co_df.drop_duplicates(
            ["mother", "father", "child", "crossover_sex", "min_pos", "max_pos"],
            inplace=True,
        )
        co_df.to_csv(output.co_filt_data, sep="\t", index=None)


rule interpolate_co_locations:
    """Interpolate the locations of crossovers from crossover specific maps."""
    input:
        co_map="results/{name}.crossover_filt.tsv.gz",
        recmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
    output:
        co_map_interp="results/{name}.crossover_filt.{recmap}.tsv.gz",
    script:
        "scripts/interp_recmap.py"


# ------- Analysis 1. Estimate Crossover Interference Stratified by Age & Sex -------- #
rule age_sex_stratified_co_interference:
    input:
        metadata=config["metadata"],
        co_map_interp="results/{name}.crossover_filt.{recmap}.tsv.gz",
        recmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
    output:
        age_sex_interference="results/{name}.age_xo_interference.{recmap}.tsv",
    params:
        nbins=10,
        nboots=5,
        seed=42,
    script:
        "scripts/est_age_strat_xo.py"


# ------- Analysis 2. Posterior estimates of CO-interference across individuals. ------- #


# ------- Analysis 3. Estimation of sex-specific recombination maps from crossover data ------ #
rule split_sex_specific_co_data:
    """Splits crossovers into maternal/paternal events."""
    input:
        co_map_interp="results/{name}.crossover_filt.{recmap}.tsv.gz",
    output:
        expand(
            "results/{{sex}}_genmap/{{name}}.events.{{recmap}}.{chrom}.{{sex}}.txt",
            chrom=chroms,
        ),
    wildcard_constraints:
        sex="maternal|paternal",
    params:
        sex=lambda wildcards: wildcards.sex,
    script:
        "scripts/gen_events.py"


rule setup_intervals_co_data:
    """Setup data for intervals on which to estimate recombination rates."""
    input:
        co_map_interp="results/{name}.crossover_filt.{recmap}.tsv.gz",
        recmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
    output:
        expand(
            "results/{{sex}}_genmap/{{name}}.nbmeioses.{{recmap}}.{chrom}.{{sex}}.txt",
            chrom=chroms,
        ),
    wildcard_constraints:
        sex="maternal|paternal",
    params:
        sex=lambda wildcards: wildcards.sex,
    script:
        "scripts/gen_nbmeioses.py"


rule est_recomb_rate_rmcmc:
    input:
        events_file="results/{sex}_genmap/{name}.events.{recmap}.{chrom}.{sex}.txt",
        nbmeioses_file="results/{sex}_genmap/{name}.nbmeioses.{recmap}.{chrom}.{sex}.txt",
        rMCMC="./rMCMC/rMCMC/rMCMC",
    output:
        rates_out="results/{sex}_genmap/{name}.{recmap}.{chrom}.{sex}-rates.txt",
        events_out="results/{sex}_genmap/{name}.{recmap}.{chrom}.{sex}-events.txt",
    params:
        outfix=lambda wildcards: "results/{wildcards.sex}_genmap/{wildcards.name}.{wildcards.recmap}.{wildcards.chrom}.{wildcards.sex}",
        nmeioses=lambda wildcards: pd.read_csv(
            f"results/{wildcards.sex}_genmap/{wildcards.name}.nbmeioses.{wildcards.recmap}.{wildcards.chrom}.{wildcards.sex}.txt",
            nrows=1,
            sep="\s",
        ).values[:, 2][0],
    shell:
        "{input.rMCMC} -i {input.events_file} -nbmeioses {input.nbmeioses_file} -m {params.nmeioses} -o {params.outfix}"
