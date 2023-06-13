#!python3


configfile: "config.yaml"


rule all:
    input:
        expand(
            "results/{name}.crossover_filt.{recmap}.tsv.gz",
            name=config["crossover_data"].keys(),
            recmap=config["recomb_maps"].keys(),
        ),


rule filter_co_dataset:
    """Filter a crossover dataset according to our key criteria."""
    input:
        crossover_data=lambda wildcards: config["crossover_data"][wildcards.name],
    output:
        co_filt_data="results/{name}.crossover_filt.tsv.gz",
    run:
        import pandas as pd

        co_df = pd.read_csv(input.crossover_data, sep="\t")
        co_df.drop_duplicates(
            ["mother", "father", "child", "crossover_sex", "min_pos", "max_pos"],
            inplace=True,
        )
        co_df.to_csv(output.co_filt_data, sep="\t", index=None)


rule interpolate_co_locations:
    """Interpolate the locations of crossovers from crossover specific maps."""
    input:
        co_map="results/{name}.crossover_filt.tsv.gz",
        recmap=lambda wildcards: config["recomb_maps"][wildcards.recmap],
    output:
        co_map_interp="results/{name}.crossover_filt.{recmap}.tsv.gz",
    script:
        "scripts/interp_recmap.py"


# ------- Analysis 1. Estimate Crossover Interference Stratified by Age & Sex -------- #
rule age_sex_stratified_co_interference:
    input:
      metadata = config["metadata"]
      co_map_interp="results/{name}.crossover_filt.{recmap}.tsv.gz",
    output:
      age_sex_interference = "results/{name}.age_xo_interference.tsv"
    params:
      nbins = 10
    script:
      "scripts/est_age_strat_xo.py"


# ------- Analysis 2. Posterior estimates of CO-interference across individuals. ------- #


# ------- Analysis 3. Estimation of sex-specific recombination maps from crossover data ------ #
