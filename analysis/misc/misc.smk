#!python3

"""Snakefile for miscellaneous analyses in the Natera dataset """

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gzip

# Create the VCF data dictionary for each chromosome ...
vcf_dict_natera_parents = {}
vcf_dict_1kg_phase3 = {}
chroms = [f"chr{i}" for i in range(1, 23)]
for c in chroms:
    vcf_dict_natera_parents[
        c
    ] = f"/data/rmccoy22/natera_spectrum/genotypes/opticall_parents_100423/genotypes/eagle_phased_hg38/natera_parents.b38.{c}.vcf.gz"
    vcf_dict_1kg_phase3[
        c
    ] = f"/scratch4/rmccoy22/sharedData/populationDatasets/1KGP_phase3/GRCh38_phased_vcfs/ALL.{c}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"


localrules:
    all,


rule all:
    input:
        "results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs.eigenvec",


# ----------- 1. PCA  of Parental Genotypes along with 1KG samples ---------- #
rule merge_1kg_phase3_natera:
    input:
        vcfgz_natera=lambda wildcards: vcf_dict_natera_parents[wildcards.chrom],
        vcfgz_1kg_phase3=lambda wildcards: vcf_dict_1kg_phase3[wildcards.chrom],
        chr_rename="data/chr_rename.txt",
    output:
        vcf_rename=temp("results/pca/kg_phase3.grch38.{chrom}.phased.snvs.vcf.gz"),
        vcf_rename_tbi=temp(
            "results/pca/kg_phase3.grch38.{chrom}.phased.snvs.vcf.gz.tbi"
        ),
        vcf_merged="results/pca/kg_phase3.natera_merged.grch38.{chrom}.phased.snvs.vcf.gz",
        vcf_merged_tbi="results/pca/kg_phase3.natera_merged.grch38.{chrom}.phased.snvs.vcf.gz.tbi",
    threads: 8
    wildcard_constraints:
        chrom="|".join(chroms),
    resources:
        time="1:00:00",
        mem_mb="4G",
    shell:
        """
        bcftools annotate --rename-chrs {input.chr_rename} {input.vcfgz_1kg_phase3} | bcftools view -v snps -c 5 -m 2 -M 2 --threads {threads} | bgzip -@{threads} > {output.vcf_rename}   
        tabix -f {output.vcf_rename}
        bcftools merge {output.vcf_rename} {input.vcfgz_natera} --threads {threads} | bcftools view -v snps -m 2 -M 2 -i 'F_MISSING < 0.01' --threads {threads} | bgzip -@{threads} > {output.vcf_merged}
        tabix -f {output.vcf_merged}
        """


rule concat_autosomes:
    input:
        merged_vcfs=expand(
            "results/pca/kg_phase3.natera_merged.grch38.{chrom}.phased.snvs.vcf.gz",
            chrom=chroms,
        ),
    output:
        concat_vcf="results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs.vcf.gz",
        concat_vcf_tbi="results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs.vcf.gz.tbi",
    threads: 12
    shell:
        "bcftools concat --threads {threads} {input.merged_vcfs} | bgzip -@{threads} > {output.concat_vcf}; tabix -f {output.concat_vcf}"


rule run_plink_pca:
    input:
        concat_vcf="results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs.vcf.gz",
        concat_vcf_tbi="results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs.vcf.gz.tbi",
    output:
        eigenvec="results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs.eigenvec",
        eigenval="results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs.eigenval",
        log="results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs.log",
    resources:
        time="1:00:00",
        mem_mb="4G",
    params:
        outfix="results/pca/kg_phase3.natera_merged.grch38.autosomes.phased.snvs",
        pcs=20,
    threads: 24
    shell:
        "plink2 --vcf {input.concat_vcf} --pca {params.pcs} approx --threads {threads} --out {params.outfix}"

