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


# ------- Rules Section ------- #
localrules:
    all,


rule all:
    input:
        #         expand(
        # "results/gwas_output/regenie/{project_name}_{sex}_{format}_{pheno}.regenie.gz",
        # project_name=config["project_name"],
        # sex=["Male", "Female"],
        # pheno=["MeanCO", "VarCO", "RandPheno"],
        # format="regenie",
        # ),
        # expand(
        # "results/gwas_output/plink2/{project_name}_{sex}_{format}.{pheno}.glm.linear",
        # format="plink2",
        # project_name=config["project_name"],
        # sex=["Male", "Female"],
        # pheno=["MeanCO", "VarCO", "RandPheno"],
        # ),
        expand(
            "results/phenotypes/{project_name}.{format}.hotspot.pheno",
            project_name=config["project_name"],
            format="plink2",
        ),


# ------- 0. Preprocess Genetic data ------- #
rule vcf2pgen:
    """Convert the VCF File into PGEN format files for use using REGENIE."""
    input:
        vcf_file=lambda wildcards: vcf_dict[wildcards.chrom],
    output:
        pgen=temp("results/pgen_input/{project_name}.{chrom}.pgen"),
        psam=temp("results/pgen_input/{project_name}.{chrom}.psam"),
        pvar=temp("results/pgen_input/{project_name}.{chrom}.pvar"),
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
        "plink2 --vcf {input.vcf_file} dosage=DS --double-id --maf 0.005 --threads {threads} --make-pgen --out {params.outfix} "


rule merge_full_pgen:
    """Merge the individual pgen files into a consolidated PGEN file."""
    input:
        pgen=expand("results/pgen_input/{{project_name}}.{chrom}.pgen", chrom=chroms),
        psam=expand("results/pgen_input/{{project_name}}.{chrom}.psam", chrom=chroms),
        pvar=expand("results/pgen_input/{{project_name}}.{chrom}.pvar", chrom=chroms),
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
        plink2 --pmerge-list {output.tmp_merge_file} --maf 0.005 --threads {threads} --make-pgen --out {params.outfix}
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
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --threads {threads} --maf 0.01 --indep-pairwise 200 25 0.4 --out {params.outfix}
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
rule create_rec_abundance_phenotypes:
    """Create the overall abundance phenotypes."""
    input:
        co_data=config["crossovers"],
    output:
        pheno="results/phenotypes/{project_name}.{format}.abundance.pheno",
    resources:
        time="0:30:00",
        mem_mb="1G",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
    script:
        "scripts/create_rec_abundance_phenotypes.py"


rule create_sex_specific_hotspots:
    """Create sex-specific hotspot files from Haldorsson et al 2019."""
    input:
        genmap=lambda wildcards: config["hotspots"][wildcards.sex],
    output:
        hotspots="results/phenotypes/appendix/{project_name}.{sex}.hotspots.tsv",
    wildcard_constraints:
        sex="Male|Female",
    params:
        srr=10,
    resources:
        time="0:10:00",
        mem_mb="4G",
    run:
        df = pd.read_csv(input.genmap, sep="\t", comment="#")
        df["SRR"] = df.cMperMb / df.cMperMb.mean()
        filt_df = df[df.SRR > params.srr]
        filt_df.rename(
            columns={"Chr": "chrom", "Begin": "start", "End": "end"}, inplace=True
        )
        filt_df.to_csv(output.hotspots, index=None, sep="\t")


rule create_hotspot_phenotypes:
    """Create phenotypes for hotspot occupancy."""
    input:
        co_data=config["crossovers"],
        pratto_hotspots=lambda wildcards: config["bed_files"]["pratto2014"],
        male_hotspots="results/phenotypes/appendix/{project_name}.Male.hotspots.tsv",
        female_hotspots="results/phenotypes/appendix/{project_name}.Female.hotspots.tsv",
    output:
        pheno="results/phenotypes/{project_name}.{format}.hotspot.pheno",
        pheno_raw="results/phenotypes/{project_name}.{format}.hotspot.raw.pheno",
    resources:
        time="2:00:00",
        mem_mb="5G",
    params:
        max_interval=50e3,
        nreps=100,
        ngridpts=300,
        plink_format=lambda wildcards: wildcards.format == "plink2",
    script:
        "scripts/create_hotspot_phenotypes.py"


rule create_rec_location_phenotypes:
    """Create the full quantitative phenotype."""
    input:
        co_data=config["crossovers"],
        centromeres=config["bed_files"]["centromeres"],
        telomeres=config["bed_files"]["telomeres"],
    output:
        pheno="results/phenotypes/{project_name}.{format}.location.pheno",
    resources:
        time="0:30:00",
        mem_mb="1G",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
    script:
        "scripts/create_rec_location_phenotypes.py"


rule combine_phenotypes:
    """Create the full quantitative phenotype."""
    input:
        abundance_pheno="results/phenotypes/{project_name}.{format}.abundance.pheno",
        location_pheno="results/phenotypes/{project_name}.{format}.location.pheno",
        hotspot_pheno="results/phenotypes/{project_name}.{format}.hotspot.pheno",
    output:
        pheno="results/phenotypes/{project_name}.{format}.pheno",
    resources:
        time="0:30:00",
        mem_mb="1G",
    wildcard_constraints:
        format="plink2|regenie",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
    run:
        abundance_df = pd.read_csv(input.abundance_pheno, sep="\t")
        location_df = pd.read_csv(input.location_pheno, sep="\t")
        hotspot_df = pd.read_csv(input.hotspot_pheno)
        pheno_df = abundance_df.merge(location_df, on=["IID"], how="outer").merge(
            hotspot_df, on=["IID"], how="outer"
        )
        pheno_df.to_csv(output.pheno, sep="\t", index=None)


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
        loco_list="results/gwas_output/regenie/predictions/{project_name}_{sex}_{format}_pred.list",
        prs_list="results/gwas_output/regenie/predictions/{project_name}_{sex}_{format}_prs.list",
    resources:
        time="6:00:00",
        mem_mb="10G",
    threads: 24
    wildcard_constraints:
        format="regenie",
    params:
        outfix=lambda wildcards: f"results/gwas_output/regenie/predictions/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}",
    shell:
        """
        regenie --step 1 --pgen results/pgen_input/{wildcards.project_name} --covarFile {input.covar} --phenoFile {input.pheno} --remove {input.sex_exclusion} --bsize 200 --apply-rint --print-prs --threads {threads} --lowmem --lowmem-prefix tmp_rg --out {params.outfix}
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
        loco_pred="results/gwas_output/regenie/predictions/{project_name}_{sex}_{format}_pred.list",
    output:
        expand(
            "results/gwas_output/{{format}}/{{project_name}}_{{sex}}_{{format}}_{pheno}.regenie.gz",
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
        regenie --step 2 --pgen results/pgen_input/{wildcards.project_name} --covarFile {input.covar} --phenoFile {input.pheno} --pred {input.loco_pred} --remove {input.sex_exclusion} --bsize 200 --apply-rint --threads {threads} --lowmem --lowmem-prefix tmp_rg --gz --out results/gwas_output/regenie/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}
        """


# ------ 4. Run GWAS using Plink2 across phenotypes ------ #
rule plink_regression:
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        pheno="results/phenotypes/{project_name}.{format}.pheno",
        covar="results/covariates/{project_name}.covars.{format}.txt",
        sex_exclusion="results/covariates/{project_name}.{sex}.{format}.exclude.txt",
    output:
        expand(
            "results/gwas_output/{{format}}/{{project_name}}_{{sex}}_{{format}}.{pheno}.glm.linear",
            pheno=["MeanCO", "VarCO", "RandPheno"],
        ),
    resources:
        time="6:00:00",
        mem_mb="10G",
    threads: 24
    wildcard_constraints:
        format="plink2",
    params:
        outfix=lambda wildcards: f"results/gwas_output/{wildcards.format}/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}",
    shell:
        "plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar}  --pheno {input.pheno} --covar {input.covar} --quantile-normalize --glm hide-covar --remove {input.sex_exclusion} --out {params.outfix}"
