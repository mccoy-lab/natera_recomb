#!python3

import numpy as np
import pandas as pd

import pickle, gzip
from tqdm import tqdm
from pathlib import Path
from io import StringIO

# ---- Parameters for inference in Natera Data ---- #
metadata_file = "../../data/spectrum_metadata_merged.csv"

# Create the VCF data dictionary for each chromosome ...
vcf_dict = {}
chroms = [f"chr{i}" for i in range(1, 23)]
for i, c in enumerate(range(1, 23)):
    vcf_dict[
        chroms[i]
    ] = f"/data/rmccoy22/natera_spectrum/genotypes/opticall_parents_031423/genotypes/eagle_phased_hg38/natera_parents.b38.chr{c}.vcf.gz"


# ------- Rules Section ------- #
localrules:
    all,


rule all:
    input:
        []


# ------- 0. Preprocess data ------- #
rule vcf2pgen:
    input:
        vcf_files = lambda wildcards: vcf_dict[wildcards.c]
    outputs:
        pgen = "results/pgen_input/{project_name}.{chrom}.pgen",
        psam = "results/pgen_input/{project_name}.{chrom}.psam",
        pvar = "results/pgen_input/{project_name}.{chrom}.pvar"
    params:
        outfix = lambda wildcards: f'results/pgen_input/{wildcards.name}.{wildcards.chrom}'
    shell:
        "plink2 --vcf {input.vcf_file} --out {params.outfix} "

