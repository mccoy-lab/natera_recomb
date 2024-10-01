# Natera Recombination Quantitative Genetics

## Pipelines

All of the pipelines are built using the `snakemake` workflow management system and held in the `analysis` directory. Each pipeline is meant to be run independently to generate intermediate results and comes with its own conda environment. Whenever possible we detail the intermdiate data generated. We briefly detail these pipelines below:

1. `analysis/crossovers`

This workflow identifies crossovers on disomic chromosomes across all embryos where there are at least three or more disomic sibling embryos (using the current `aneuploidy` calls - see below). 

Briefly the workflow for each chromosome operates to:

* refine the phase of parental haplotypes using the multi-sibling embryo data under a model of disomy
* estimate apparent crossovers using a BAF-derived likelihood ratio extension to the model of Coop et al 2007 

2. `analysis/co_post_process`

3. `analysis/simulations/`

4. `analysis/gwas`: Conducts full sex-specific GWAS and heritability analysis for recombination-derived phenotypes. 

### Important Data Tables

The overall pipeline relies on a number of data tables to perform inference properly:

1. Aneuploidy data table: `/data/rmccoy22/natera_spectrum/karyohmm_outputs/compiled_output/natera_embryos.karyohmm_v30a.bph_sph_trisomy.full_annotation.031624.tsv.gz`

This table details the output of the MetaHMM model from `karyohmm`, which returns chromosome-wide posterior probabilities of a chromosome being in a disomic or non-disomic state. The fields are better described in the `natera_aneuploidy` project.

2. Crossover Table (Raw): `/scratch16/rmccoy22/abiddan1/natera_recomb/analysis/crossovers/results/test_crossover_heuristic.v30b.nsib_support.geno_qual.tsv.gz`

This table contains raw output of crossover calls from the pipeline (without any filtering based on local ploidy estimation) 

3. Crossover Table (Filtered): `/scratch16/rmccoy22/abiddan1/natera_recomb/analysis/co_post_process/results/v30b_heuristic_90_nsib_qual.crossover_filt.deCode_haldorsson19.merged.meta.tsv.gz` 

This table contains an intersection with the Haldorsson et al data as well (to interpolate the genetic map lengths). The filters applied are:

* The minimum and maximum positions in the window must have > 90% marginal posterior probability of being disomic (mean across sibling embryos)
* Duplicate windows have been removed in this case
* Intersection with Natera metadata for parental age association.

## Contact

@aabiddanda or abiddan1@jhu.edu
