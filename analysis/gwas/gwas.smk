#!python3

import numpy as np
import pandas as pd

import pickle, gzip
from tqdm import tqdm
from io import StringIO


# ---- Parameters for inference in Natera Data ---- #
# configfile: "config.yaml"


# Create the VCF data dictionary for each chromosome ...
vcf_dict = {}
chroms = [f"chr{i}" for i in range(1, 24)]
autosomes = [f"chr{i}" for i in range(1, 23)]
for chrom in chroms:
    vcf_dict[chrom] = (
        f"{config['datadir']}spectrum_imputed_{chrom}_rehead_filter_cpra.vcf.gz"
    )


# ------- Defining the phenotypes to explore ------ #
phenotypes = [
    "RandMeanCO",
    "MeanCO",
    "VarCO",
    "cvCO",
    "RandPheno",
    "CentromereDist",
    "TelomereDist",
    "HotspotOccupancy",
    "GcContent",
    "ReplicationTiming",
]


# ------- Rules Section ------- #
localrules:
    all,


rule all:
    input:
        expand(
            "results/gwas_output/{format}/finalized/{project_name}.sumstats.replication.rsids.tsv",
            format=["plink2", "regenie"],
            project_name=config["project_name"],
        ),
        # expand(
        # "results/h2/h2sq_ldms/h2_est_total/{project_name}.total.hsq",
        # project_name=config["project_name"],
        # ),
        # expand(
        # "results/h2/h2sq_chrom/h2_est_total/{project_name}.total.hsq",
        # project_name=config["project_name"],
        # ),


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
        mem_mb="10G",
    threads: 12
    params:
        outfix=lambda wildcards: f"results/pgen_input/{wildcards.project_name}.{wildcards.chrom}",
    shell:
        """
        plink2 --vcf {input.vcf_file} --double-id \
        --max-alleles 2 --maf 0.005 --mac 1 --memory 9000\
        --threads {threads} --make-pgen --lax-chrx-import --out {params.outfix} 
        """


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
        mem_mb="12G",
    threads: 12
    shell:
        """
        for i in {chroms}; do echo \"results/pgen_input/{wildcards.project_name}.$i\" ; done > {output.tmp_merge_file}
        plink2 --pmerge-list {output.tmp_merge_file} --maf 0.005 --threads {threads} --memory 10000 --make-pgen --out {params.outfix}
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
    threads: 12
    params:
        npcs=20,
        outfix=lambda wildcards: f"results/covariates/{wildcards['project_name']}",
    shell:
        """
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --threads {threads} --maf 0.01 --memory 9000 --indep-pairwise 200 25 0.4 --out {params.outfix}
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --extract {output.keep_variants} --memory 9000 --pca {params.npcs} approx --threads {threads} --out {params.outfix}
        """


rule king_related_individuals:
    """Isolate related individuals (up to 2nd degree) to remove from downstream GWAS analyses."""
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        keep_variants="results/covariates/{project_name}.prune.in",
    output:
        king_includes=temp("results/covariates/{project_name}.king.cutoff.in.id"),
        king_excludes="results/covariates/{project_name}.king.cutoff.out.id",
    resources:
        time="1:00:00",
        mem_mb="10G",
    threads: 20
    params:
        outfix=lambda wildcards: f"results/covariates/{wildcards['project_name']}",
        king_thresh=0.125,
    shell:
        """
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --extract {input.keep_variants} --threads {threads} --memory 9000 --maf 0.01 --king-cutoff {params.king_thresh} --out {params.outfix}
        """


rule create_full_covariates:
    """Create the full set of covariates to use downstream GWAS applications."""
    input:
        co_data=config["crossovers"],
        evecs="results/covariates/{project_name}.eigenvec",
        metadata=config["metadata"],
    output:
        covars="results/covariates/{project_name}.covars.{format}.txt",
    wildcard_constraints:
        format="regenie|plink2",
    resources:
        time="1:00:00",
        mem_mb="8G",
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
        time="1:00:00",
        mem_mb="8G",
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
        mem_mb="8G",
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
        replication_timing=config["bed_files"]["replication_timing"],
        gc_content=config["bed_files"]["gc_content"],
    output:
        pheno="results/phenotypes/{project_name}.{format}.location.pheno",
    resources:
        time="1:00:00",
        mem_mb="8G",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
        gc_window=250,
    script:
        "scripts/create_rec_location_phenotypes.py"


rule combine_phenotypes:
    """Create the full quantitative phenotype."""
    input:
        abundance_pheno="results/phenotypes/{project_name}.{format}.abundance.pheno",
        location_pheno="results/phenotypes/{project_name}.{format}.location.pheno",
        hotspot_pheno="results/phenotypes/{project_name}.{format}.hotspot.pheno",
    output:
        pheno="results/phenotypes/{project_name}.{format}.raw.pheno",
    resources:
        time="1:00:00",
        mem_mb="8G",
    wildcard_constraints:
        format="plink2|regenie",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
        outlier_sd=3,
    run:
        abundance_df = pd.read_csv(input.abundance_pheno, sep="\t")
        location_df = pd.read_csv(input.location_pheno, sep="\t")
        hotspot_df = pd.read_csv(input.hotspot_pheno, sep="\t")
        if params.plink_format:
            merge1_df = abundance_df.merge(location_df, on=["#FID", "IID"], how="outer")
            pheno_df = merge1_df.merge(hotspot_df, on=["#FID", "IID"], how="outer")
            pheno_df.drop_duplicates(subset=["IID"], inplace=True)
            for k in pheno_df.columns:
                if k not in ["#FID", "IID"]:
                    x = pheno_df[k].values
                    mu, sd = np.nanmean(x), np.nanstd(x)
                    pheno_df[k] = pheno_df[k].where(
                        (mu - params.outlier_sd * sd <= pheno_df[k])
                        & (pheno_df[k] <= mu + params.outlier_sd * sd)
                    )
            pheno_df.to_csv(output.pheno, sep="\t", na_rep="NA", index=None)
        else:
            merge1_df = abundance_df.merge(location_df, on=["FID", "IID"], how="outer")
            pheno_df = merge1_df.merge(hotspot_df, on=["FID", "IID"], how="outer")
            pheno_df.drop_duplicates(subset=["IID"], inplace=True)
            for k in pheno_df.columns:
                if k not in ["FID", "IID"]:
                    x = pheno_df[k].values
                    mu, sd = np.nanmean(x), np.nanstd(x)
                    pheno_df[k] = pheno_df[k].where(
                        (mu - params.outlier_sd * sd <= pheno_df[k])
                        & (pheno_df[k] <= mu + params.outlier_sd * sd)
                    )
            pheno_df.to_csv(output.pheno, sep="\t", na_rep="NA", index=None)


rule within_sex_rint:
    """Apply within-sex IRNT transformations for quantitative phenotypes."""
    input:
        pheno="results/phenotypes/{project_name}.{format}.raw.pheno",
        covar="results/covariates/{project_name}.covars.{format}.txt",
    output:
        pheno="results/phenotypes/{project_name}.{format}.pheno",
    wildcard_constraints:
        format="plink2|regenie",
    script:
        "scripts/sex_specific_irnt.py"


# ------ 3. Run GWAS using REGENIE across phenotypes ------ #
rule create_sex_exclude_file:
    input:
        covar="results/covariates/{project_name}.covars.{format}.txt",
        king_excludes="results/covariates/{project_name}.king.cutoff.out.id",
    output:
        sex_specific="results/covariates/{project_name}.{sex}.{format}.exclude.txt",
    wildcard_constraints:
        sex="Male|Female|Joint",
    resources:
        time="1:00:00",
        mem_mb="8G",
    params:
        plink_format=lambda wildcards: wildcards.format == "plink2",
    run:
        cov_df = pd.read_csv(input["covar"], sep="\t")
        king_df = pd.read_csv(input["king_excludes"], sep="\t")
        if wildcards.sex == "Male":
            exclude_sex_df = cov_df[cov_df.Sex == 0]
        if wildcards.sex == "Female":
            exclude_sex_df = cov_df[cov_df.Sex == 1]
        if wildcards.sex == "Joint":
            exclude_sex_df = cov_df[cov_df.Sex == 2]
        if not params["plink_format"]:
            king_df.columns = ["FID", "IID"]
        concat_df = pd.concat([exclude_sex_df, king_df])
        if not params["plink_format"]:
            out_df = concat_df[["FID", "IID"]].drop_duplicates(subset=["IID"])
        else:
            out_df = concat_df[["#FID", "IID"]].drop_duplicates(subset=["IID"])
        out_df.to_csv(output["sex_specific"], index=None, sep="\t")


# -------- GWAS Steps in REGENIE ---------- #
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
        include_snps="results/gwas_output/regenie/predictions/{project_name}_{sex}_{format}.prune.in",
        exclude_snps=temp(
            "results/gwas_output/regenie/predictions/{project_name}_{sex}_{format}.prune.out"
        ),
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
        plink2 --pfile results/pgen_input/{wildcards.project_name} --threads {threads} --autosome --maf 0.005 --mac 5 --memory 9000 --indep-pairwise 200 25 0.4 --out {params.outfix}
        regenie --step 1 --pgen results/pgen_input/{wildcards.project_name} --extract {output.include_snps} --covarFile {input.covar} --phenoFile {input.pheno} --remove {input.sex_exclusion} --bsize 200 --apply-rint --print-prs --threads {threads} --lowmem --lowmem-prefix tmp_rg --out {params.outfix}
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
            pheno=phenotypes,
        ),
    resources:
        time="6:00:00",
        mem_mb="10G",
    threads: 16
    wildcard_constraints:
        format="regenie",
    shell:
        """
        regenie --step 2 --pgen results/pgen_input/{wildcards.project_name} --covarFile {input.covar} --phenoFile {input.pheno} --pred {input.loco_pred} --remove {input.sex_exclusion} --bsize 200 --apply-rint --threads {threads} --lowmem --lowmem-prefix tmp_rg --gz --out results/gwas_output/regenie/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}
        """


rule convert2plink_format:
    """Convert regenie marginal summary stats to plink format for clumping and loci identification."""
    input:
        regenie="results/gwas_output/{format}/{project_name}_{sex}_{format}_{pheno}.regenie.gz",
    output:
        regenie2plink="results/gwas_output/{format}/{project_name}_{sex}_{format}.{pheno}.glm.linear",
    wildcard_constraints:
        format="regenie",
    resources:
        time="1:00:00",
        mem_mb="10G",
    shell:
        """
       zcat {input} | awk \'NR==1 {{print "#CHROM\tPOS\tID\tREF\tALT\tA1\tA1_FREQ\tTEST\tOBS_CT\tBETA\tSE\tT_STAT\tP"}} NR > 1 {{OFS="\t"; print $1,$2,$3,$4,$5,$5,$6,$8,$7,$9,$10,$11,10**(-$12)}}\' > {output.regenie2plink}
       """


# ------ 4. Run GWAS using Plink2 across phenotypes w. GC correction. ------ #
rule plink_regression:
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        pheno="results/phenotypes/{project_name}.{format}.pheno",
        covar="results/covariates/{project_name}.covars.{format}.txt",
        sex_exclusion="results/covariates/{project_name}.{sex}.{format}.exclude.txt",
    output:
        gwas_sumstats=expand(
            "results/gwas_output/{{format}}/{{project_name}}_{{sex}}_{{format}}.raw.{pheno}.glm.linear",
            pheno=phenotypes,
        ),
        gwas_sumstats_adjusted=expand(
            "results/gwas_output/{{format}}/{{project_name}}_{{sex}}_{{format}}.raw.{pheno}.glm.linear.adjusted",
            pheno=phenotypes,
        ),
    resources:
        time="6:00:00",
        mem_mb="10G",
    threads: 16
    wildcard_constraints:
        format="plink2",
    params:
        outfix=lambda wildcards: f"results/gwas_output/{wildcards.format}/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}.raw",
    shell:
        """
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar}\
        --pheno {input.pheno} --covar {input.covar} --threads {threads}\
        --memory 9000 --covar-quantile-normalize --glm hide-covar\
        --adjust gc --remove {input.sex_exclusion} --out {params.outfix}
        """


rule match_pval_gc:
    """Match the p-values from the adjusted setting for GC control."""
    input:
        gwas_results="results/gwas_output/{format}/{project_name}_{sex}_{format}.raw.{pheno}.glm.linear",
        gwas_adjusted="results/gwas_output/{format}/{project_name}_{sex}_{format}.raw.{pheno}.glm.linear.adjusted",
    output:
        gwas_results="results/gwas_output/{format}/{project_name}_{sex}_{format}.{pheno}.glm.linear",
    wildcard_constraints:
        format="plink2",
    resources:
        time="1:00:00",
        mem_mb="5G",
    shell:
        "awk 'FNR == NR && NR > 1 {{a[$2]=$5; next}} {{if ($3 in a) {{$15=a[$3]; print $0}} else {{print $0}}}}' {input.gwas_adjusted} {input.gwas_results} | awk 'NR > 1' | tr [:blank:] '\t'  > {output.gwas_results}"


rule plink_clumping:
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        sex_exclusion="results/covariates/{project_name}.{sex}.{format}.exclude.txt",
        gwas_results="results/gwas_output/{format}/{project_name}_{sex}_{format}.{pheno}.glm.linear",
    output:
        "results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.clumps",
    resources:
        time="1:00:00",
        mem_mb="10G",
    threads: 8
    params:
        outfix=lambda wildcards: f"results/gwas_output/{wildcards.format}/clumped/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}.{wildcards.pheno}",
        pval=1e-5,
    shell:
        """
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar}\
        --threads {threads} --memory 9000 --clump-unphased --clump {input.gwas_results}\
        --clump-r2 0.1 --clump-kb 1000 --remove {input.sex_exclusion}\
        --clump-p1 {params.pval} --out {params.outfix}
        """


rule obtain_effect_sizes:
    """Obtain effect sizes for lead variants in a clump."""
    input:
        clumps="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.clumps",
        gwas_results="results/gwas_output/{format}/{project_name}_{sex}_{format}.{pheno}.glm.linear",
    output:
        header=temp(
            "results/gwas_output/{format}/{project_name}_{sex}_{format}.{pheno}.glm.header"
        ),
        temp_var_ids=temp(
            "results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.tmp.var_ids"
        ),
        temp_top_variants=temp(
            "results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.tmp.top_vars"
        ),
        top_variants="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.top_vars",
    shell:
        """
        awk 'NR == 1 {{print $0}}' {input.gwas_results} | sed s'/#//g' > {output.header}
        awk 'NR > 1 {{print $3}}' {input.clumps} > {output.temp_var_ids} 
        awk 'FNR == NR {{a[$1]++; next}} {{if ($3 in a) {{print $0}}}}' {output.temp_var_ids} {input.gwas_results} > {output.temp_top_variants} 
        cat {output.header} {output.temp_top_variants} > {output.top_variants}
        """


rule obtain_allele_frequencies:
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        sex_exclusion="results/covariates/{project_name}.{sex}.{format}.exclude.txt",
        clumps="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.clumps",
    output:
        temp_vars=temp(
            "results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.freqs.tmpvars"
        ),
        freqs="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.afreq",
    params:
        outfix=lambda wildcards: f"results/gwas_output/{wildcards.format}/clumped/{wildcards.project_name}_{wildcards.sex}_{wildcards.format}.{wildcards.pheno}",
    threads: 4
    shell:
        """
        awk 'NR > 1 {{print $3}}' {input.clumps} > {output.temp_vars}
        plink2 --pgen {input.pgen} --psam {input.psam} --pvar {input.pvar} --threads {threads} --extract {output.temp_vars} --remove {input.sex_exclusion} --freq --out {params.outfix}
        """


# ----------- Mapping variants to genes ----------- #
rule reformat_gencode_bed:
    input:
        gencode_annotation=config["gencode"],
    output:
        "results/resources/gencode.bed",
    shell:
        'zcat {input.gencode_annotation} | grep -v "^#" | awk \'$3 == "gene"\' | awk -F"\t" \'{{print $1"\t"$4"\t"$5"\t"$9}}\' | grep "protein" | bedtools sort > {output}'


rule map_snp2gene:
    """Mapping the lead SNP of a cluster to the nearest protein coding gene in gencode."""
    input:
        clump_results="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.clumps",
        gencode="results/resources/gencode.bed",
    output:
        clump_bed=temp(
            "results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.snp2gene.clump.bed"
        ),
        reformat_clumps=temp(
            "results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.snp2gene.clump.reform"
        ),
        snp2gene=temp(
            "results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.snp2gene"
        ),
        reform_sumstats="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.sumstats.tsv",
    shell:
        """
        awk '{{print $3}}' {input.clump_results} | grep -v \"ID\" |  awk -F \":\" \'{{print $1\"\t\"$2\"\t\"$2}}\' | bedtools sort > {output.clump_bed}
        bedtools closest -a {output.clump_bed} -b {input.gencode} -k 1 -d > {output.snp2gene}
        awk \'NR > 1 {{OFS=\"\t\"; $1=\"chr\"$1; print $0}}\' {input.clump_results} | sort -k1,2n > {output.reformat_clumps}
        awk \'FNR==NR {{a[$1,$2]=$0; next}} {{print a[$1,$2],$0}}\' {output.reformat_clumps} {output.snp2gene} | sort -k4 -g > {output.reform_sumstats}
        """


rule combine_gwas_effect_size_afreq:
    input:
        sumstats="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.sumstats.tsv",
        freqs="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.afreq",
        top_variants="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.top_vars",
    output:
        final_sumstats="results/gwas_output/{format}/clumped/{project_name}_{sex}_{format}.{pheno}.sumstats.final.tsv",
    resources:
        time="1:00:00",
        mem_mb="8G",
    script:
        "scripts/combine_freq_clump_plink.py"


rule combine_gwas_results:
    """Combine GWAS results across """
    input:
        sumstats=expand(
            "results/gwas_output/{{format}}/clumped/{{project_name}}_{sex}_{{format}}.{pheno}.sumstats.final.tsv",
            pheno=phenotypes,
            sex=["Male", "Female", "Joint"],
        ),
    output:
        sumstats_final=temp(
            "results/gwas_output/{format}/finalized/{project_name}.sumstats.tsv"
        ),
    resources:
        time="1:00:00",
        mem_mb="8G",
    run:
        tot_dfs = []
        for fp in input.sumstats:
            df = pd.read_csv(fp, sep="\t")
            tot_dfs.append(df)
        final_df = pd.concat(tot_dfs)
        final_df.to_csv(output.sumstats_final, sep="\t", index=None)


rule intersect_w_replication_data:
    """Intersect with replication dataset from Haldorsson 2019."""
    input:
        sumstats="results/gwas_output/{format}/finalized/{project_name}.sumstats.tsv",
        replication=config["gwas_replication"],
    output:
        sumstats_replication=temp(
            "results/gwas_output/{format}/finalized/{project_name}.sumstats.replication.tsv"
        ),
    script:
        "scripts/gwas_replication.py"


rule add_rsids:
    """Add rsids for lead variants."""
    input:
        sumstats_replication="results/gwas_output/{format}/finalized/{project_name}.sumstats.replication.tsv",
        dbsnp=config["dbsnp"],
    output:
        temp_regions=temp(
            "results/gwas_output/{format}/finalized/{project_name}.sumstats.replication.rsids.tmp_regions.txt"
        ),
        temp_rsids=temp(
            "results/gwas_output/{format}/finalized/{project_name}.sumstats.replication.rsids.tmp_rsids.txt"
        ),
        sumstats_rsids="results/gwas_output/{format}/finalized/{project_name}.sumstats.replication.rsids.tsv",
    shell:
        """
        awk \'NR > 1 {{print $2}}\' {input.sumstats_replication} | awk  -F \":\" \'{{print $1\"\t\"$2}}\' | sed \'s/^chr//g\' > {output.temp_regions}
        tabix -R {output.temp_regions} {input.dbsnp} | awk \'NR == 1 {{print \"ID\tRSID\"}} {{split($5, a, \",\"); for (x in a) {{print \"chr\"$1\":\"$2\":\"$4\":\"a[x]\"\t\"$3}}}}\' > {output.temp_rsids}
        awk \'FNR == NR {{a[$1] = $2; next}} {{print $0\"\t\"a[$2]}}\' {output.temp_rsids} {input.sumstats_replication} > {output.sumstats_rsids}
        """


# -------- 5. Estimating per-chromosome h2 using GREML -------- #
rule estimate_per_chrom_sex_grm:
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        excludes="results/covariates/{project_name}.{sex}.plink2.exclude.txt",
    output:
        grm="results/h2/h2sq_chrom/grms/{project_name}.{sex}.{chrom}.grm.bin",
        grm_n="results/h2/h2sq_chrom/grms/{project_name}.{sex}.{chrom}.grm.N.bin",
        grm_id="results/h2/h2sq_chrom/grms/{project_name}.{sex}.{chrom}.grm.id",
    params:
        chrom=lambda wildcards: f"{wildcards.chrom}"[3:],
        prefix=lambda wildcards: f"results/pgen_input/{wildcards.project_name}",
        outfix=lambda wildcards: f"results/h2/h2sq_chrom/grms/{wildcards.project_name}.{wildcards.sex}.{wildcards.chrom}",
    resources:
        time="4:00:00",
        mem_mb="10G",
    threads: 8
    shell:
        """
        gcta --pfile {params.prefix}\
        --chr {params.chrom}\
        --remove {input.excludes}\
        --make-grm\
        --out {params.outfix}\
        --threads {threads}
        """


rule create_gcta_pheno:
    """Create phenotype file for use in GCTA."""
    input:
        pheno="results/phenotypes/{project_name}.plink2.pheno",
    output:
        pheno=temp(
            "results/h2/h2sq_chrom/pheno/{project_name}.{sex}.{chrom}.{pheno}.txt"
        ),
    resources:
        time="0:10:00",
        mem_mb="2G",
    run:
        df = pd.read_csv(input.pheno, sep="\t")
        filt_df = df[["#FID", "IID", f"{wildcards.pheno}"]]
        filt_df.to_csv(output.pheno, sep="\t", na_rep="NA", index=None, header=None)


rule create_gcta_covar:
    """Create a GCTA-structured covariate file."""
    input:
        covar="results/covariates/{project_name}.covars.plink2.txt",
    output:
        covar="results/h2/h2sq_chrom/covars/{project_name}.{sex}.{chrom}.{pheno}.covars.txt",
    resources:
        time="0:10:00",
        mem_mb="2G",
    run:
        df = pd.read_csv(input.covar, sep="\t")
        assert "Sex" in df.columns
        df.drop(["Sex"], axis=1, inplace=True)
        df.to_csv(output.covar, sep="\t", na_rep="NA", index=None)


# Need a rule to create phenotypes for GCTA + covariates ...
rule per_chrom_reml:
    input:
        grm="results/h2/h2sq_chrom/grms/{project_name}.{sex}.{chrom}.grm.bin",
        pheno="results/h2/h2sq_chrom/pheno/{project_name}.{sex}.{chrom}.{pheno}.txt",
        covar="results/h2/h2sq_chrom/covars/{project_name}.{sex}.{chrom}.{pheno}.covars.txt",
    output:
        hsq="results/h2/h2sq_chrom/h2_est/{project_name}.{sex}.{chrom}.{pheno}.hsq",
    params:
        grmfix=lambda wildcards: f"results/h2/h2sq_chrom/grms/{wildcards.project_name}.{wildcards.sex}.{wildcards.chrom}",
        outfix=lambda wildcards: f"results/h2/h2sq_chrom/h2_est/{wildcards.project_name}.{wildcards.sex}.{wildcards.chrom}.{wildcards.pheno}",
    resources:
        time="2:00:00",
        mem_mb="10G",
    threads: 4
    shell:
        """
        gcta --reml --grm {params.grmfix}\
        --pheno {input.pheno} --qcovar {input.covar}\
        --out {params.outfix} --threads {threads}
        """


rule collapse_per_chrom_h2:
    """Collapse the per-chromosome estimates for h2."""
    input:
        hsq_files=expand(
            "results/h2/h2sq_chrom/h2_est/{{project_name}}.{{sex}}.{chrom}.{{pheno}}.hsq",
            chrom=autosomes,
        ),
    output:
        h2sq_tsv="results/h2/h2sq_chrom/h2_est_total/{project_name}.{sex}.{pheno}.hsq",
    run:
        dfs = []
        for fp in input.hsq_files:
            chrom = fp.split(".")[2]
            df = pd.read_csv(fp, nrows=4, sep="\t")
            df["chrom"] = chrom
            df["sex"] = f"{wildcards.sex}"
            df["pheno"] = f"{wildcards.pheno}"
            dfs.append(df)
        tot_df = pd.concat(dfs)
        tot_df.to_csv(output.h2sq_tsv, index=None, sep="\t")


rule aggregate_per_chrom_h2_multitrait:
    input:
        hsq_files=expand(
            "results/h2/h2sq_chrom/h2_est_total/{{project_name}}.{sex}.{pheno}.hsq",
            sex=["Male", "Female"],
            pheno=phenotypes,
        ),
    output:
        tot_hsq="results/h2/h2sq_chrom/h2_est_total/{project_name}.total.hsq",
    run:
        tot_df = pd.concat([pd.read_csv(fp, sep="\t") for fp in input.hsq_files])
        tot_df.to_csv(output.tot_hsq, sep="\t", index=None)


# -------- 6. Heritability Estimation using GREML-LDMS from imputed data -------- #
rule estimate_ld_scores:
    """Estimate the LD-scores per-variant using GCTA."""
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        king_excludes="results/covariates/{project_name}.king.cutoff.out.id",
    output:
        bed=temp("results/h2/h2sq_ldms/ld_score/{project_name}.{chrom}.bed"),
        bim=temp("results/h2/h2sq_ldms/ld_score/{project_name}.{chrom}.bim"),
        fam=temp("results/h2/h2sq_ldms/ld_score/{project_name}.{chrom}.fam"),
        ldscore="results/h2/h2sq_ldms/ld_score/{project_name}.{chrom}.score.ld",
    params:
        chrom=lambda wildcards: f"{wildcards.chrom}"[3:],
        ldscore_region=1000,
        pfile=lambda wildcards: f"results/pgen_input/{wildcards.project_name}",
        outbfix=lambda wildcards: f"results/h2/h2sq_ldms/ld_score/{wildcards.project_name}.{wildcards.chrom}",
        outfix=lambda wildcards: f"results/h2/h2sq_ldms/ld_score/{wildcards.project_name}.{wildcards.chrom}",
    resources:
        time="4:00:00",
        mem_mb="20G",
    threads: 8
    shell:
        """
        plink2 --pfile {params.pfile} --chr {params.chrom} --make-bed --out {params.outbfix}
        gcta --bfile {params.outbfix}\
        --threads {threads} --chr {params.chrom}\
        --remove {input.king_excludes}\
        --ld-score-region {params.ldscore_region}\
        --out {params.outfix}
        """


rule partition_ld_scores:
    """Partition LD scores into multiple components."""
    input:
        expand(
            "results/h2/h2sq_ldms/ld_score/{{project_name}}.{chrom}.score.ld",
            chrom=autosomes,
        ),
    output:
        ld_maf_partition="results/h2/h2sq_ldms/ld_score/{project_name}.ld_{p}.maf_{i}.txt",
    resources:
        time="1:00:00",
        mem_mb="4G",
    params:
        nld_bins=config["h2"]["ld_bins"],
        nmaf_bins=config["h2"]["maf_bins"],
        maf_bins=[0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    script:
        "scripts/partition_ld_scores.py"


rule create_grms:
    """Create a LD + MAF stratified GRM for estimation of effects."""
    input:
        pgen="results/pgen_input/{project_name}.pgen",
        psam="results/pgen_input/{project_name}.psam",
        pvar="results/pgen_input/{project_name}.pvar",
        excludes="results/covariates/{project_name}.{sex}.plink2.exclude.txt",
        ldms_snps="results/h2/h2sq_ldms/ld_score/{project_name}.ld_{p}.maf_{i}.txt",
    output:
        grm="results/h2/h2sq_ldms/grms/{project_name}.{sex}.ld_{p}.maf_{i}.grm.bin",
        grm_n="results/h2/h2sq_ldms/grms/{project_name}.{sex}.ld_{p}.maf_{i}.grm.N.bin",
        grm_id="results/h2/h2sq_ldms/grms/{project_name}.{sex}.ld_{p}.maf_{i}.grm.id",
    params:
        pfile=lambda wildcards: f"results/pgen_input/{wildcards.project_name}",
        outfix=lambda wildcards: f"results/h2/h2sq_ldms/grms/{wildcards.project_name}.{wildcards.sex}.ld_{wildcards.p}.maf_{wildcards.i}",
    resources:
        time="4:00:00",
        mem_mb="20G",
    threads: 4
    shell:
        """
        gcta --pfile {params.pfile}\
        --threads {threads}\
        --remove {input.excludes}\
        --extract {input.ldms_snps}\
        --make-grm\
        --out {params.outfix}
        """


rule estimate_h2_ldms:
    """Estimate h2snp stratified by LD + MAF."""
    input:
        grms=expand(
            "results/h2/h2sq_ldms/grms/{{project_name}}.{{sex}}.ld_{p}.maf_{i}.grm.bin",
            p=range(config["h2"]["ld_bins"]),
            i=range(config["h2"]["maf_bins"]),
        ),
        pheno="results/h2/h2sq_chrom/pheno/{project_name}.{sex}.chr1.{pheno}.txt",
        covar="results/h2/h2sq_chrom/covars/{project_name}.{sex}.chr1.{pheno}.covars.txt",
    output:
        mgrms=temp("results/h2sq_ldms/h2_est/{project_name}.{sex}.{pheno}.mgrms"),
        hsq="results/h2/h2sq_ldms/h2_est/{project_name}.{sex}.{pheno}.hsq",
    params:
        outfix=lambda wildcards: f"results/h2/h2sq_ldms/h2_est/{wildcards.project_name}.{wildcards.sex}.{wildcards.pheno}",
    resources:
        time="4:00:00",
        mem_mb="10G",
    threads: 8
    shell:
        """
        ls {input.grms} | sed 's/\.grm.*//' > {output.mgrms}
        gcta --reml --mgrm {output.mgrms} --pheno {input.pheno} --qcovar {input.covar} --reml-no-constrain --out {params.outfix} --threads {threads}
        """


rule collapse_per_chrom_h2_ldms:
    """Collapse the estimates for h2."""
    input:
        hsq="results/h2/h2sq_ldms/h2_est/{project_name}.{sex}.{pheno}.hsq",
    output:
        h2sq_tsv="results/h2/h2sq_ldms/h2_est_total/{project_name}.{sex}.{pheno}.hsq",
    run:
        df = pd.read_csv(snakemake.input["hsq"], sep="\t", on_bad_lines="skip")
        df["sex"] = f"{wildcards.sex}"
        df["pheno"] = f"{wildcards.pheno}"
        df.to_csv(output.h2sq_tsv, index=None, sep="\t")


rule aggregate_h2_ldms:
    """Final aggregation rule for estimating h2 in GREML-LDMS."""
    input:
        hsq_files=expand(
            "results/h2/h2sq_ldms/h2_est_total/{{project_name}}.{sex}.{pheno}.hsq",
            sex=["Male", "Female"],
            pheno=phenotypes,
        ),
    output:
        tot_hsq="results/h2/h2sq_ldms/h2_est_total/{project_name}.total.hsq",
    run:
        tot_df = pd.concat([pd.read_csv(fp, sep="\t") for fp in input.hsq_files])
        tot_df.to_csv(output.tot_hsq, sep="\t", index=None)
