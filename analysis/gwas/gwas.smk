#!python3

import numpy as np
import pandas as pd

import pickle, gzip
from tqdm import tqdm
from pathlib import Path
from io import StringIO


# ---- Parameters for inference in Natera Data ---- #
metadata_file = "../../data/spectrum_metadata_merged.csv"

configfile: "config.yaml"

# Create the VCF data dictionary for each chromosome ...
vcf_dict = {}
chroms = [f"chr{i}" for i in range(1, 23)]
for i, c in enumerate(range(1, 23)):
    vcf_dict[
        chroms[i]
    ] = f"/data/rmccoy22/natera_spectrum/genotypes/imputed_parents_031423/spectrum_imputed_chr{c}_rehead_filter.vcf.gz"
    
# f"/data/rmccoy22/natera_spectrum/genotypes/opticall_parents_031423/genotypes/eagle_phased_hg38/natera_parents.b38.chr{c}.vcf.gz"


# ------- Rules Section ------- #
localrules:
    all,


rule all:
    input:
        expand("results/pgen_input/{project_name}.pgen", project_name=config[''])


# ------- 0. Preprocess Genetic data ------- #
rule vcf2pgen:
    """Convert the VCF File into PGEN format files for use using REGENIE."""
    input:
        vcf_files = lambda wildcards: vcf_dict[wildcards.chrom]
    outputs:
        pgen = "results/pgen_input/{project_name}.{chrom}.pgen",
        psam = "results/pgen_input/{project_name}.{chrom}.psam",
        pvar = "results/pgen_input/{project_name}.{chrom}.pvar"
    params:
        outfix = lambda wildcards: f'results/pgen_input/{wildcards.name}.{wildcards.chrom}'
    shell:
        "plink2 --vcf {input.vcf_file} dosage=DS --double-id --make-pgen --out {params.outfix} "


rule merge_full_pgen:
    """Merge the individual pgen files into a consolidated PGEN file."""
    input:
        pgen = expand("results/pgen_input/{{project_name}}.{chrom}.pgen", chrom=chroms)
    output:
        tmp_merge_file = temp("results/pgen_input/{project_name}.txt"),
        pgen = "results/pgen_input/{project_name}.pgen",
        psam = "results/pgen_input/{project_name}.psam",
        pvar = "results/pgen_input/{project_name}.pvar"
    params:
        outfix = lambda wildcards: f'results/pgen_input/{wildcards.name}'
    shell:
        """
        for i in {chroms}; echo \"results/pgen_input/{wildcards.project_name}.$i\" ; done > {output.tmp_merge_file}
        plink2 --pmerge-list {output.tmp_merge_file} --make-pgen --out {params.outfix}
        """


# ------- 1. Prepare the key covariate file ------- #



# ------ 2. Run GWAS using REGENIE across phenotypes ------ #
