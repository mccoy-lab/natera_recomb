#!python3

import numpy as np
import pandas as pd

import pickle, gzip
from tqdm import tqdm
from pathlib import Path
from io import StringIO


# ---- Parameters for inference in Natera Data ---- #
configfile: "config.yaml"


include: "snakefiles/phenotyping.smk"


# Create the VCF data dictionary for each chromosome ...
vcf_dict = {}
chroms = [f"chr{i}" for i in range(19, 23)]
for chrom in chroms:
    vcf_dict[
        chrom
    ] = f"{config['datadir']}spectrum_imputed_{chrom}_rehead_filter.vcf.gz"

# f"/data/rmccoy22/natera_spectrum/genotypes/opticall_parents_031423/genotypes/eagle_phased_hg38/natera_parents.b38.chr{c}.vcf.gz"


# ------- Rules Section ------- #
localrules:
    all,


rule all:
    input:
        expand(
            "results/covariates/{project_name}.eigenvec",
            project_name=config["project_name"],
        ),
        expand(
            "results/covariates/{project_name}.king.cutoff.out.id",
            project_name=config["project_name"],
        ),


# ------- 0. Preprocess Genetic data ------- #
rule vcf2pgen:
    """Convert the VCF File into PGEN format files for use using REGENIE."""
    input:
        vcf_file=lambda wildcards: vcf_dict[wildcards.chrom],
    output:
        pgen="results/pgen_input/{project_name}.{chrom}.pgen",
        psam="results/pgen_input/{project_name}.{chrom}.psam",
        pvar="results/pgen_input/{project_name}.{chrom}.pvar",
    wildcard_constraints:
        chrom="|".join(chroms),
        project_name=config["project_name"],
    params:
        outfix=lambda wildcards: f"results/pgen_input/{wildcards.project_name}.{wildcards.chrom}",
    shell:
        "plink2 --vcf {input.vcf_file} dosage=DS --double-id --make-pgen --out {params.outfix} "


rule merge_full_pgen:
    """Merge the individual pgen files into a consolidated PGEN file."""
    input:
        pgen=expand("results/pgen_input/{{project_name}}.{chrom}.pgen", chrom=chroms),
    output:
        tmp_merge_file=temp("results/pgen_input/{project_name}.txt"),
        merge_pgen=temp("results/pgen_input/{project_name}-merge.pgen"),
        merge_psam=temp("results/pgen_input/{project_name}-merge.psam"),
        merge_pvar=temp("results/pgen_input/{project_name}-merge.pvar"),
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
    params:
        outfix=lambda wildcards: f"results/pgen_input/{wildcards['project_name']}",
    shell:
        """
        for i in {chroms}; do echo \"results/pgen_input/{wildcards.project_name}.$i\" ; done > {output.tmp_merge_file}
        plink2 --pmerge-list {output.tmp_merge_file} --make-pgen --out {params.outfix}
        """


# ------- 1. Prepare the covariate files + list of samples for analyses ------- #
rule compute_pcs:
    """Compute PCA using genotype data."""
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
    output:
        keep_variants="results/covariates/{project_name}.prune.in",
        remove_variants=temp("results/covariates/{project_name}.prune.out"),
        evecs="results/covariates/{project_name}.eigenvec",
        evals="results/covariates/{project_name}.eigenval",
    params:
        npcs=20,
        outfix=lambda wildcards: f"results/covariates/{wildcards['project_name']}",
    shell:
        """
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --maf 0.01 --indep-pairwise 200 25 0.2 --out {params.outfix}
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --extract {output.keep_variants} --pca {params.npcs} approx --out {params.outfix}
        """


rule king_related_individuals:
    """Isolate related individuals (up to 2nd degree) to remove from downstream GWAS analyses."""
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
    output:
        king_includes=temp("results/covariates/{project_name}.king.cutoff.in.id"),
        king_excludes="results/covariates/{project_name}.king.cutoff.out.id",
    params:
        outfix=lambda wildcards: f"results/covariates/{wildcards['project_name']}",
        king_thresh=0.125,
    shell:
        """
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --maf 0.05 --king-cutoff {params.king_thresh} --out {params.outfix}
        """


rule create_full_covariates:
    """Create the full set of covariates to use downstream GWAS applications."""
    input:
        evecs="results/covariates/{project_name}.eigenvec",
        metadata=config["metadata"],
    output:
        "results/covariates/{project_name}.covars.{format}.txt",
    wildcard_constraints:
        format="regenie|plink2",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
    script:
        "scripts/combine_covariates.py"


# ------ 2. Create the underlying phenotypes ------ #
rule create_full_quant_phenotypes:
    """Create the full quantitative phenotype."""
    input:
        co_master_folder=config["crossovers"],
    output:
        pheno="results/phenotypes/{project_name}.pheno",
    params:
        phenotype_type="quant",
    script:
        "scripts/create_rec_phenotypes.py"


# ------ 3. Run GWAS using REGENIE across phenotypes ------ #


# ------ 4. Run GWAS using Plink2 across phenotypes ------ #
