library(susieR)
library(data.table)
library(dplyr)
library(glue)
library(tibble)

create_loci <- function(locus_sumstats_df, trait, window_size = 500000, pval = 1e-8) {
  # Filter to just the locus focused on this trait
  sumstats_filt_df <- locus_sumstats_df %>% filter(PHENO == trait, P < pval)
  if (nrow(sumstats_filt_df) > 0){
    sumstats_filt_df <- sumstats_filt_df %>%
      rowwise() %>%
      mutate(CHROM = strsplit(ID, ":")[[1]][1], POS = as.numeric(strsplit(ID, ":")[[1]][2])) %>%
      arrange(P)
  }
  regions <- c()
  while (nrow(sumstats_filt_df) > 0) {
    chrom <- sumstats_filt_df$CHROM[1]
    if (chrom != "chrX"){
      pos <- as.integer(sumstats_filt_df$POS[1])
      gene <- sumstats_filt_df$Gene[1]
      regions <- c(regions, glue("{chrom}:{pos-window_size}-{pos+window_size}:{gene}"))
      sumstats_filt_df <- sumstats_filt_df %>% filter(CHROM != chrom, !between(POS, (pos - window_size), (pos + window_size))) %>% arrange(P)
    }
  }
  return(regions)
}


subset_sumstats_ld_matrix <- function(sumstats_df, chrom = "chr5", start = NA, end = NA, threads=12, outfix = "/tmp/test1") {
  subset_plink <- glue("plink2 --pgen {snakemake@input[['pgen']]} --pvar {snakemake@input['pvar']} --psam {snakemake@input['psam']} --remove {snakemake@input[['sex_exclude']]} --chr {chrom} --from-bp {start} --to-bp {end} --make-bed --out {outfix} --threads {threads}")
  system(subset_plink)
  ldmatrix <- glue("plink --bfile {outfix} --keep-allele-order  --r square --threads {threads} --out {outfix}")
  system(ldmatrix)
  R_hat <- as.matrix(fread(glue("{outfix}.ld")))
  sumstats_filt_df <- sumstats_df %>% filter(CHROM == chrom, POS > start, POS < end)
  n <- sumstats_filt_df$OBS_CT[1]
  if (nrow(sumstats_filt_df) != dim(R_hat)[1]) {
    stop("Sumstats and LD-matrix are mis-aligned!")
  }
  betas <- sumstats_filt_df$BETA
  ses <- sumstats_filt_df$SE
  system(glue("rm {outfix}*"))
  return(list(betas = betas, ses = ses, n = n, R = R_hat, sumstats = sumstats_filt_df))
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
  snp_ids <- sumstats_filt_cs_df$ID
  results <- data.frame(ID = character(), pip = numeric(), credible_set = character())
  for (variant_idx in 1:length(snp_ids)) {
    results[nrow(results) + 1, ] <- c(snp_ids[variant_idx], susie_res$pip[variant_idx], NA)
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


# Read in the summary stat of interest & annotate with the trait of interest
sumstats_df <- fread(snakemake@input[["raw_sumstats"]], nThread=snakemake@threads[[1]])
loci_df <- fread(snakemake@input[["locus_sumstats"]], nThread=snakemake@threads[[1]])
trait <- glue("{snakemake@wildcards[['pheno']]}_{snakemake@wildcards[['sex']]}")
sumstats_df <- sumstats_df %>% rowwise() %>%  mutate(CHROM = strsplit(ID, ":")[[1]][1])

# Running Susie on each locus
locus_finemapped_sumstats <- list()
loci <- create_loci(loci_df, trait = trait)
print(loci)
i <- 1
for (region in loci) {
  region_str <- strsplit(region, ":|-")[[1]]
  chrom <- region_str[1]
  start <- as.integer(region_str[2])
  end <- as.integer(region_str[3])
  gene <- region_str[4]
  regionfix <- gsub(":|-", "_", region)
  print(region_str)
  outfix = glue('/tmp/tmp_{regionfix}')
  subset_res <- subset_sumstats_ld_matrix(sumstats_df, chrom = chrom, start = start, end = end, outfix=outfix, threads=snakemake@threads[[1]])
  susie_res_df <- run_susie(subset_res)
  susie_res_df$region <- region
  susie_res_df$Gene <- gene
  locus_finemapped_sumstats[[i]] <- susie_res_df
  i <- i + 1
}

# Collapse the different genome-wide significant-loci
full_finemapped_df <- bind_rows(locus_finemapped_sumstats)
full_finemapped_df$PHENO <- trait
write.table(full_finemapped_df, snakemake@output[["finemapped_sumstats"]], append = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
