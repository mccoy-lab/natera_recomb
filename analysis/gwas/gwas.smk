#!python3

import numpy as np
import pandas as pd

import pickle, gzip
from tqdm import tqdm
from pathlib import Path
from io import StringIO


# ---- Parameters for inference in Natera Data ---- #
configfile: "config.yaml"


# Create the VCF data dictionary for each chromosome ...
vcf_dict = {}
chroms = [f"chr{i}" for i in range(20, 23)]
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
            "results/gwas_output/regenie/{project_name}_{sex}_{format}_{pheno}.regenie.gz",
            project_name=config["project_name"],
            sex=["Male", "Female"],
            pheno=["MeanCO", "VarCO", "RandPheno"],
            format="regenie",
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
    resources:
        time="1:00:00",
        mem_mb="8G",
    threads: 24
    params:
        outfix=lambda wildcards: f"results/pgen_input/{wildcards.project_name}.{wildcards.chrom}",
    shell:
        "plink2 --vcf {input.vcf_file} dosage=DS --double-id --maf 0.001 --threads {threads} --make-pgen --out {params.outfix} "


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
    resources:
        time="1:00:00",
        mem_mb="5G",
    threads: 24
    shell:
        """
        for i in {chroms}; do echo \"results/pgen_input/{wildcards.project_name}.$i\" ; done > {output.tmp_merge_file}
        plink2 --pmerge-list {output.tmp_merge_file} --maf 0.001 --threads {threads} --make-pgen --out {params.outfix}
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
    resources:
        time="1:00:00",
        mem_mb="10G",
    threads: 24
    params:
        npcs=20,
        outfix=lambda wildcards: f"results/covariates/{wildcards['project_name']}",
    shell:
        """
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --threads {threads} --maf 0.01 --indep-pairwise 200 25 0.2 --out {params.outfix}
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --extract {output.keep_variants} --pca {params.npcs} approx --threads {threads} --out {params.outfix}
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
    resources:
        time="1:00:00",
        mem_mb="1G",
    threads: 20
    params:
        outfix=lambda wildcards: f"results/covariates/{wildcards['project_name']}",
        king_thresh=0.125,
    shell:
        """
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --threads {threads} --maf 0.01 --king-cutoff {params.king_thresh} --out {params.outfix}
        """


rule create_full_covariates:
    """Create the full set of covariates to use downstream GWAS applications."""
    input:
        evecs="results/covariates/{project_name}.eigenvec",
        metadata=config["metadata"],
    output:
        covars="results/covariates/{project_name}.covars.{format}.txt",
    wildcard_constraints:
        format="regenie|plink2",
    resources:
        time="0:30:00",
        mem_mb="1G",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
    script:
        "scripts/combine_covariates.py"


# ------ 2. Create the underlying phenotypes ------ #
rule create_full_quant_phenotypes:
    """Create the full quantitative phenotype."""
    input:
        co_data=config["crossovers"],
    output:
        pheno="results/phenotypes/{project_name}.{format}.pheno",
    resources:
        time="0:30:00",
        mem_mb="1G",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
    script:
        "scripts/create_rec_phenotypes.py"


# ------ 3. Run GWAS using REGENIE across phenotypes ------ #
rule create_sex_exclude_file:
    input:
        covar="results/covariates/{project_name}.covars.{format}.txt",
        king_excludes="results/covariates/{project_name}.king.cutoff.out.id",
    output:
        sex_specific="results/covariates/{project_name}.{sex}.{format}.exclude.txt",
    wildcard_constraints:
        sex="Male|Female",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
    run:
        cov_df = pd.read_csv(input["covar"], sep="\t")
        king_df = pd.read_csv(input["king_excludes"], sep="\t")
        if wildcards.sex == "Male":
            exclude_sex_df = cov_df[cov_df.Sex == 0]
        else:
            exclude_sex_df = cov_df[cov_df.Sex == 1]
        if ~params["plink_format"]:
            king_df.columns = ["FID", "IID"]
        concat_df = pd.concat([exclude_sex_df, king_df])
        if ~params["plink_format"]:
            out_df = concat_df[["FID", "IID"]].drop_duplicates()
        else:
            out_df = concat_df[["#FID", "IID"]].drop_duplicates()
        out_df.to_csv(output["sex_specific"], index=None, sep="\t")


rule regenie_step1:
    """Run the first step of REGENIE for polygenic prediction."""
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        pheno="results/phenotypes/{project_name}.{format}.pheno",
        covar="results/covariates/{project_name}.covars.{format}.txt",
        sex_exclusion="results/covariates/{project_name}.{sex}.{format}.exclude.txt",
    output:
        loco_list="results/gwas_output/regenie/{project_name}_{sex}_{format}.list",
        prs_list="results/gwas_output/regenie/{project_name}_{sex}_{format}_prs.list",
    resources:
        time="6:00:00",
        mem_mb="10G",
    threads: 24
    wildcard_constraints:
        format="regenie",
    shell:
        """
        regenie --step 1 --pgen results/pgen_input/{wildcards.project_name} --covarFile {input.covar} --phenoFile {input.pheno} --remove {input.sex_exclusion} --bsize 200 --apply-rint --print-prs --threads {threads} --lowmem --lowmem-prefix tmp_rg --out results/gwas_output/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}
        """


rule regenie_step2:
    """Run the second step of REGENIE for effect-size estimation."""
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        pheno="results/phenotypes/{project_name}.{format}.pheno",
        covar="results/covariates/{project_name}.covars.{format}.txt",
        sex_exclusion="results/covariates/{project_name}.{sex}.{format}.exclude.txt",
        loco_pred="results/gwas_output/regenie/{project_name}_{sex}_{format}.list",
    output:
        expand(
            "results/gwas_output/regenie/{{project_name}}_{{sex}}_{{format}}_{pheno}.regenie.gz",
            pheno=["MeanCO", "VarCO", "RandPheno"],
        ),
    resources:
        time="6:00:00",
        mem_mb="10G",
    threads: 24
    wildcard_constraints:
        format="regenie",
    shell:
        """
        regenie --step 2 --pgen results/pgen_input/{wildcards.project_name} --covarFile {input.covar} --phenoFile {input.pheno} --pred {input.loco_pred} --remove {input.sex_exclusion} --bsize 200 --apply-rint --threads {threads} --lowmem --lowmem-prefix tmp_rg --gz --out results/gwas_output/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}
        """


# ------ 4. Run GWAS using Plink2 across phenotypes ------ #
