library(data.table)
library(tidyverse)
library(lme4)
library(performance)
library(cowplot)
library(emmeans)
library(marginaleffects)
library(pbapply)
library(lmerTest)

# ------------- Model 1: Maternal Crossover Counts ------------- #

dt <- fread("../../analysis/co_post_process/results/v30b_heuristic_90_nsib_qual.crossover_filt.deCode_haldorsson19.crossover_count.maternal.euploid.csv.gz") %>%
  .[, is_aneuploid_embryo := FALSE]

dt2 <- fread("../../analysis/co_post_process/results/v30b_heuristic_90_nsib_qual.crossover_filt.deCode_haldorsson19.crossover_count.maternal.aneuploid.csv.gz") %>%
  .[, is_aneuploid_embryo := TRUE]

dt <- rbind(dt, dt2) %>%
  .[bf_max_cat == "2"]

dt[, AVGSIGMA := weighted.mean(sigma_baf, cM_len), by = uid]
dt[, AVGPI0 := weighted.mean(pi0_baf, cM_len), by = uid]

dt[egg_donor == 1, patient_age := as.numeric(25)]
# NOTE: remove sperm and egg donor individuals from modeling here ... 
df <- dt %>% filter((egg_donor == 0) & (sperm_donor == 0))

print(df %>% group_by(is_aneuploid_embryo) %>% summarize(avg=mean(nco), n=n(), sd=sd(nco), se=sd/sqrt(n)))

m1_co <- glmer(data = df, 
            formula = nco ~ (1 | IID / uid) + scale(patient_age) + PC1 + PC2 + PC3 + PC4 + PC5 + 
              PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 +
              PC18 + PC19 + PC20 + offset(log(cM_len)) + scale(AVGSIGMA) + scale(AVGPI0) + scale(NEMBRYO) + is_aneuploid_embryo,
            family = poisson,
            nAGQ = 0,
            control = glmerControl(optimizer = "bobyqa"))

# Summarize the model ... 
summary(m1_co)
emmeans(m1_co, "is_aneuploid_embryo", type = "response")
summary(m1_co)$coefficients[,4]

# ------------- Model 2: Maternal Crossover Hotspot Occupancy ------------- #
dt <- fread("../../analysis/co_post_process/results/v30b_heuristic_90_nsib_qual.crossover_filt.deCode_haldorsson19.hotspot_occupy.maternal.euploid.csv.gz") %>%
  .[, is_aneuploid_embryo := FALSE]

dt2 <- fread("../../analysis/co_post_process/results/v30b_heuristic_90_nsib_qual.crossover_filt.deCode_haldorsson19.hotspot_occupy.maternal.aneuploid.csv.gz") %>%
  .[, is_aneuploid_embryo := TRUE]

dt <- rbind(dt, dt2)

# dt[, AVGSIGMA := weighted.mean(sigma_baf, cM_len), by = uid]
# dt[, AVGPI0 := weighted.mean(pi0_baf, cM_len), by = uid]

# dt[egg_donor == 1, patient_age := as.numeric(25)]

# This just kind of filters out the zeros and 1s?
df <- dt %>% filter(nchrom >= 17)
head(df)

print(df %>% group_by(is_aneuploid_embryo) %>% summarize(avg=mean(mean_alpha_mat), median=median(mean_alpha_mat), n=n(), sd=sd(mean_alpha_mat), se=sd/sqrt(n)))

m1_hotspot <- glmer(data = df, 
            formula = mean_alpha_mat ~ (1 | IID) + scale(Age) + PC1 + PC2 + PC3 + PC4 + PC5 + 
              PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 +
              PC18 + PC19 + PC20 + scale(AVGSIGMA) + scale(AVGPI0) + scale(NEMBRYO) + is_aneuploid_embryo,
            family=Gamma(link = "log"),
            nAGQ = 0,
            control = glmerControl(optimizer = "bobyqa"))
summary(m1_hotspot)
emmeans(m1_hotspot, "is_aneuploid_embryo", type = "response")

# --------------- Model 3: Maternal Centromere + Telomere Distances ---------------- #
dt <- fread("../../analysis/co_post_process/results/v30b_heuristic_90_nsib_qual.crossover_filt.deCode_haldorsson19.centromere_telomere_dist.maternal.euploid.csv.gz") %>%
  .[, is_aneuploid_embryo := FALSE]

dt2 <- fread("../../analysis/co_post_process/results/v30b_heuristic_90_nsib_qual.crossover_filt.deCode_haldorsson19.centromere_telomere_dist.maternal.aneuploid.csv.gz") %>%
  .[, is_aneuploid_embryo := TRUE]

dt <- rbind(dt, dt2)

df <- dt %>% filter((egg_donor == 0) & (sperm_donor == 0))
head(df)


print(df %>% group_by(is_aneuploid_embryo) %>% summarize(avg=mean(centromere_dist), median=median(centromere_dist), n=n(), sd=sd(centromere_dist), se=sd/sqrt(n)))
print(df %>% group_by(is_aneuploid_embryo) %>% summarize(avg=mean(telomere_dist), median=median(telomere_dist), n=n(), sd=sd(telomere_dist), se=sd/sqrt(n)))

m1_centro <- lmer(data = df, 
            formula = centromere_dist ~ (1 | mother/uid) + scale(Age) + PC1 + PC2 + PC3 + PC4 + PC5 + 
              PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 +
              PC18 + PC19 + PC20 + scale(AVGSIGMA) + scale(AVGPI0) + scale(NEMBRYO) + is_aneuploid_embryo,
            control = lmerControl(optimizer = "bobyqa"))
summary(m1_centro)
emmeans(m1_centro, "is_aneuploid_embryo", type = "response")
coef(summary(as(m1_centro,"merModLmerTest")))