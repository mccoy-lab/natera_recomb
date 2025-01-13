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
            "results/{name}/{sex}_genmap/{name}.{chrom}.{sex}.{raw}-rates.txt",
            name=config["crossover_callset"].keys(),
            sex=["maternal", "paternal"],
            chrom=chroms,
            raw="raw",
        ),


# ------- Analysis: Estimation of sex-specific recombination maps from crossover data ------ #
rule split_sex_specific_co_data:
    """Splits crossovers into maternal/paternal events."""
    input:
        co_map=lambda wildcards: config["crossover_callset"][wildcards.name],
    output:
        events="results/{name}/{sex}_genmap/{name}.events.{chrom}.{sex}.txt",
    wildcard_constraints:
        sex="maternal|paternal",
        chrom="|".join(chroms),
    resources:
        mem_mb="10G",
    params:
        sex=lambda wildcards: wildcards.sex,
    script:
        "scripts/gen_events.py"


rule setup_intervals_co_data:
    """Setup intervals on which to estimate recombination rates."""
    input:
        co_map=lambda wildcards: config["crossover_callset"][wildcards.name],
    output:
        nbmeioses="results/{name}/{sex}_genmap/{name}.nbmeioses.{chrom}.{sex}.{raw}.txt",
    wildcard_constraints:
        sex="maternal|paternal",
        raw="raw|split",
    resources:
        mem_mb="10G",
    params:
        sex=lambda wildcards: wildcards.sex,
        nsplit=5,
        use_raw=lambda wildcards: wildcards.raw == "raw",
    script:
        "scripts/gen_nbmeioses.py"


rule est_recomb_rate_rmcmc:
    input:
        events_file="results/{name}/{sex}_genmap/{name}.events.{chrom}.{sex}.txt",
        nbmeioses_file="results/{name}/{sex}_genmap/{name}.nbmeioses.{chrom}.{sex}.{raw}.txt",
        rMCMC="./rMCMC/rMCMC/rMCMC",
    output:
        rates_out="results/{name}/{sex}_genmap/{name}.{chrom}.{sex}.{raw}-rates.txt",
        events_out="results/{name}/{sex}_genmap/{name}.{chrom}.{sex}.{raw}-events.txt",
    resources:
        time="2:00:00",
        mem_mb="5G",
    params:
        outfix=lambda wildcards: f"results/{wildcards.name}/{wildcards.sex}_genmap/{wildcards.name}.{wildcards.chrom}.{wildcards.sex}.{wildcards.raw}",
        nmeioses=lambda wildcards: pd.read_csv(
            f"results/{wildcards.name}/{wildcards.sex}_genmap/{wildcards.name}.nbmeioses.{wildcards.chrom}.{wildcards.sex}.{wildcards.raw}.txt",
            nrows=1,
            sep="\s",
        ).values[:, 2][0],
    shell:
        "{input.rMCMC} -i {input.events_file} -nbmeioses {input.nbmeioses_file} -m {params.nmeioses} -o {params.outfix}"

rule identify_mismatches:
    """Isolate places in the map that are different between the sexes."""
    input:
       maternal_rates = "results/{name}/maternal_genmap/{name}.{chrom}.maternal.{raw}-rates.txt", 
       paternal_rates = "results/{name}/paternal_genmap/{name}.{chrom}.paternal.{raw}-rates.txt", 
    output:
       differences = "results/{name}/genmap_diffs/{name}.{chrom}.{raw}-diff.txt" 
    script:
        "scripts/isolate_differences.py"
