#!python3

configfile: config.yaml

rule all:
    input:
        []

rule estimate_gamma_by_age:
    """Estimate parameters of the """
    input:
        crossover_calls = config['co_calls']
    output:
        "results/co_by_age/co_by_age_est_gamma.{sex}.tsv"
    wildcard_constraints:
        sex="maternal|paternal"
    script:
        "scripts/co_inf_by_age.R"


rule estimate_lint_by_age:
    input:
       
    output:
    script:
        "scripts/estimate_lint.py"

