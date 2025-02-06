library(susieR)
library(fread)
library(dplyr)
library(glue)


create_loci <- function(sumstats_df, window_size=1e6){
  # Create loci for 

}


subset_sumstats_ld_matrix <- function(sumstats_df, chrom='chr5', start=NA, end=NA, outfix="/tmp/test1"){
  subset_plink <- glue("plink2 --pgen {snakemake@input[['pgen']]} --pvar {snakemake@input['pvar']} --psam {snakemake@input['psam']} --remove {snakemake. --chr {chrom} --from-bp {start} --to-bp {end} --make-bed --out {outfix} --threads 12")
  system(subset_plink)
  ldmatrix <- glue("plink --bfile {outfix}  --r2 square --out {outfix}")
  system(ldmatrix)
  R_hat <- as.matrix(fread(glue('{outfix}.ld'))
  sumstats_filt_df <- sumstats_df %>% filter((POS >= start) & (POS <= end) & (CHROM == chrom))
  n <- sumstats_filt_df$OBS_CT[1]
  if(nrow(sumstats_filt_df) != dim(R_hat)[1]){
    stop("Sumstats and LD-matrix are mis-aligned!")   
  }
  betas <- sumstats_filt_df$BETA
  ses <- sumstats_filt_df$SE
  system(glue("rm {outfix}*"))
  return(list(beta=betas, se=ses, n=n, R=R_hat, sumstats=sumstats_filt_df))
}

run_susie <- function(res){
 susie_res <- susie_rss(
    beta=res$betas,
    se=res$ses
    n=res$n,
    R=res$R,
    L=10,
    coverage=0.95,
    min_abs_corr = 0.5,
    estimate_residual_variance=TRUE
 )
 # NOTE: now annotate the downstream summary stats with the credible set annotation & PIP annotation
 sumstats_filt_cs_df <- res$sumstats
 sumstats_filt_cs_df$PIP <- susie_res$pip
 return(sumstats_filt_cs_df)
}


# Read in the summary stats for the trait of interest
sumstats_df <- fread(snakemake@input[['sumstats_raw']])

loci <- create_loci(sumstats_df)
collapsed_sumstats = c()
for (region in loci){
  region_str = strsplit(region, ":|-")[[1]]

}

