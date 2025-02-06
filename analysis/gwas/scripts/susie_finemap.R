library(susieR)
library(fread)
library(dplyr)
library(glue)


create_loci <- function(sumstats_df, window_size = 1e6) {
  # Create loci for independent loci
}


subset_sumstats_ld_matrix <- function(sumstats_df, chrom = "chr5", start = NA, end = NA, outfix = "/tmp/test1") {
  subset_plink <- glue("plink2 --pgen {snakemake@input[['pgen']]} --pvar {snakemake@input['pvar']} --psam {snakemake@input['psam']} --remove {snakemake. --chr {chrom} --from-bp {start} --to-bp {end} --make-bed --out {outfix} --threads 12")
  system(subset_plink)
  ldmatrix <- glue("plink --bfile {outfix} --keep-allele-order  --r square --out {outfix}")
  system(ldmatrix)
  R_hat <- as.matrix(fread(glue("{outfix}.ld")))
  sumstats_filt_df <- sumstats_df %>% filter((POS >= start) & (POS <= end) & (CHROM == chrom))
  n <- sumstats_filt_df$OBS_CT[1]
  if (nrow(sumstats_filt_df) != dim(R_hat)[1]) {
    stop("Sumstats and LD-matrix are mis-aligned!")
  }
  betas <- sumstats_filt_df$BETA
  ses <- sumstats_filt_df$SE
  system(glue("rm {outfix}*"))
  return(list(beta = betas, se = ses, n = n, R = R_hat, sumstats = sumstats_filt_df))
}

run_susie <- function(res) {
  susie_res <- susie_rss(
    bhat = res$betas,
    shat = res$ses,
    n = res$n,
    R = res$R,
    L = 10
  )
  # NOTE: now annotate the downstream summary stats with credible set annotations & PIP annotation
  sumstats_filt_cs_df <- res$sumstats
  sumstats_filt_cs_df$PIP <- susie_res$pip
  snp_ids <- sumstats_filt_df$ID
  results <- data.frame(variant_id = character(), pip = numeric(), credible_set = character())
  for (variant_idx in 1:length(snp_ids)) {
    results[nrow(results) + 1, ] <- c(snp_ids[variant_idx], res_prdm9$pip[variant_idx], NA)
  }
  cs_results <- data.frame(credible_set = character(), cs_nvars = integer(), cs_coverage = numeric(), cs_min_corr = numeric(), cs_mean_corr = numeric(), cs_median_corr = numeric())

  if (length(names(susie_res$set$cs)) > 0) {
    for (i in 1:length(names(susie_res$set$cs))) {
      cred_set <- names(susie_res$set$cs)[i]
      cs_coverage <- susie_res$set$coverage[i]
      cs_nvars <- length(susie_res$set$cs[[i]])
      min_corr <- susie_res$set$purity[cred_set, "min.abs.corr"]
      mean_corr <- susie_res$set$purity[cred_set, "mean.abs.corr"]
      median_corr <- susie_res$set$purity[cred_set, "median.abs.corr"]
      cs_results[nrow(cs_results) + 1, ] <- c(cred_set, cs_nvars, cs_coverage, min_corr, mean_corr, median_corr)
      for (var_index in susie_res$set$cs[cred_set][[1]]) {
        results[var_index, "credible_set"] <- cred_set
      }
    }
  }
  sumstats_filt_cs_df <- sumstats_filt_cs_df %>% merge(., results, by = "ID")
  return(sumstats_filt_cs_df)
}


# Read in the summary stats for the trait of interest
sumstats_df <- fread(snakemake@input[["sumstats_raw"]])
sumstats_df <- sumstats_df %>% mutate(CHROM = paste("chr", "#CHROM", sep = ""))

locus_finemapped_sumstats <- list()
loci <- create_loci(sumstats_df)
i <- 1
for (region in loci) {
  region_str <- strsplit(region, ":|-")[[1]]
  chrom <- region_str[1]
  start <- region_str[2]
  end <- region_str[3]
  subset_res <- subset_sumstats_ld_matrix(sumstats_df, chrom = chrom, start = start, end = end)
  susie_res_df <- run_susie(subset_res)
  susie_res_df$region <- region_str
  locus_finemapped_sumstats[[i]] <- susie_res_df
  i <- i + 1
}

# Collapse the different genome-wide significant-loci
full_finemapped_df <- bind_rows(locus_finemapped_sumstats)
write.table(full_finemapped_df, snakemake@output[["finemapped_sumstats"]], append = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
