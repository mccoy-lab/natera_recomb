library(lme4)
library(data.table)
library(dplyr)

# Setup the main arguments 
args = commandArgs(trailingOnly=TRUE)

if (length(args) < 3) {
  stop("At least one argument must be supplied!", call.=FALSE)
} 


co_df = fread(args[1], sep="\t")

bootvar <- function(data, i) {
  y <- data[i]
  var(y)
}

# Make the table of the mean effects per-chromosome
per_chrom_mean_effect_df = data.frame()
per_chrom_var_effect_df = data.frame()


# Now we run maternal and paternal crossover fidelity 
chroms = unique(co_df$chrom)
for (c in chroms){
  # Get the maternal crossovers on each chromosome
  test_chr_maternal_df = co_df %>% 
    filter(chrom == c & crossover_sex == "maternal") %>% 
    group_by(uid) %>% 
    summarize(chrom=first(chrom), child=first(child), mother=first(mother), nco=n(), age = mean(patient_age), aneuploid=first(aneuploid)) 
  
  # Get the paternal crossovers on each chromosome
  test_chr_paternal_df = co_df %>% 
    filter(chrom == c & crossover_sex == "paternal") %>% 
    group_by(uid) %>% 
    summarize(chrom=first(chrom), child=first(child), father=first(father), nco=n(), age = mean(patient_age), aneuploid=first(aneuploid))
  
  model_mat_real = glmer("nco ~ age + aneuploid + (1|mother)", data = test_chr_maternal_df, family = 'poisson')
  model_mat_null = glmer("nco ~ age + (1|mother)", data = test_chr_maternal_df, family = 'poisson')
  a_mat_res = anova(model_mat_null, model_mat_real)
  
  model_pat_real = glmer("nco ~ age + aneuploid + (1|father)", data = test_chr_paternal_df, family = 'poisson')
  model_pat_null = glmer("nco ~ age + (1|father)", data = test_chr_paternal_df, family = 'poisson')
  a_pat_res = anova(model_pat_null, model_pat_real)
  
  # Create the resultant dataframes ... 
  maternal_effect_df = cbind(chrom=c(c,c,c), sex="maternal", chisq_anova =a_mat_res$Chisq[2],  p_anova=a_mat_res$`Pr(>Chisq)`[2], summary(model_mat_real)$coefficients)
  paternal_effect_df = cbind(chrom=c(c,c,c), sex="paternal", chisq_anova =a_pat_res$Chisq[2],  p_anova=a_pat_res$`Pr(>Chisq)`[2],  summary(model_pat_real)$coefficients) 
  maternal_effect_df = cbind(variable = rownames(maternal_effect_df), maternal_effect_df)
  paternal_effect_df = cbind(variable = rownames(paternal_effect_df), paternal_effect_df)
  rownames(maternal_effect_df) <- NULL
  rownames(paternal_effect_df) <- NULL
  
  per_chrom_mean_effect_df = rbind(per_chrom_mean_effect_df, maternal_effect_df, paternal_effect_df)
  
  # Look at the variance effects
  mat_eu_df = test_chr_maternal_df %>% filter(!aneuploid)
  mat_aneu_df = test_chr_maternal_df %>% filter(aneuploid)
  pat_eu_df = test_chr_maternal_df %>% filter(!aneuploid)
  pat_aneu_df = test_chr_maternal_df %>% filter(aneuploid)
  
  eu_var_mat = boot::boot(mat_eu_df$nco, bootvar, R=500)
  aneu_var_mat = boot::boot(mat_eu_df$nco, bootvar, R=500)
  eu_var_pat = boot::boot(pat_eu_df$nco, bootvar, R=500)
  aneu_var_pat = boot::boot(pat_aneu_df$nco, bootvar, R=500)
 
  # Append the output to a resulting data frame (NOTE: we should have a test in here somewhere I think?)
  per_chrom_var_effect_df = rbind(per_chrom_var_effect_df, c(c, "maternal", eu_var_mat$t0, sd(eu_var_mat$t), nrow(mat_eu_df), aneu_var_mat$t0, sd(aneu_var_mat$t0), nrow(mat_aneu_df)))
  per_chrom_var_effect_df = rbind(per_chrom_var_effect_df, c(c, "paternal", eu_var_pat$t0, sd(eu_var_pat$t), nrow(pat_eu_df), aneu_var_pat$t0, sd(aneu_var_pat$t0), nrow(pat_aneu_df)))
  cat("Finished", c, "\n")
}

# Write out the fields here to be used ... 
data.table::fwrite(per_chrom_mean_effect_df, file=arg[2], sep="\t")

colnames(per_chrom_var_effect_df) <- c("chrom", "sex", "var_co_eu", "sd_var_co_eu", "n_aneu", "var_co_aneu", "sd_var_co_aneu", "n_aneu")
data.table::fwrite(per_chrom_var_effect_df, file=arg[3], sep="\t")

