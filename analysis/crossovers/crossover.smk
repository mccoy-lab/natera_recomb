#!python3

import numpy as np
import pandas as pd

import pickle, gzip
from tqdm import tqdm
from pathlib import Path
from io import StringIO

# ---- Parameters for inference in Natera Data ---- #
metadata_file = "../../data/spectrum_metadata_merged.csv"
aneuploidy_calls = "/data/rmccoy22/natera_spectrum/karyohmm_outputs/compiled_output/natera_embryos.karyohmm_v11.052723.tsv.gz"

# Create the VCF data dictionary for each chromosome ...
vcf_dict = {}
chroms = [f"chr{i}" for i in range(1, 23)]
for i, c in enumerate(range(1, 23)):
    vcf_dict[
        chroms[i]
    ] = f"/data/rmccoy22/natera_spectrum/genotypes/opticall_parents_031423/genotypes/eagle_phased_hg38/natera_parents.b38.chr{c}.vcf.gz"

# Read in the aggregate metadata file
meta_df = pd.read_csv(metadata_file)


def find_child_data(
    child_id, meta_dfi=meta_df, raw_data_path="/data/rmccoy22/natera_spectrum/data/"
):
    """Find the child csv file based on the provided meta_df."""
    child_year = meta_df[meta_df.array == child_id].year.values
    for year in child_year:
        child_fp = Path(f"{raw_data_path}{year}/{child_id}.csv.gz")
        if child_fp.is_file():
            return child_fp, True
        else:
            continue
    return None, False


def create_trios(
    meta_df, sample_file, raw_data_path="/data/rmccoy22/natera_spectrum/data/"
):
    """Create a list of valid trio datasets."""
    valid_trios = []
    unique_mothers = np.unique(
        meta_df[meta_df.family_position == "mother"].array.values
    )
    for m in tqdm(unique_mothers):
        cases = np.unique(meta_df[meta_df.array == m].casefile_id.values)
        cur_df = meta_df[np.isin(meta_df.casefile_id, cases)]
        fathers = np.unique(cur_df[cur_df.family_position == "father"].array.values)
        if fathers.size > 1:
            print(f"More than one partner for {m}")
            for fat in fathers:
                cur_cases = np.unique(meta_df[meta_df.array == fat].casefile_id.values)
                cur_df = meta_df[np.isin(meta_df.casefile_id, cur_cases)]
                for c in cur_df[cur_df.family_position == "child"].array.values:
                    valid_trios.append((m, fat, c))
        elif fathers.size == 1:
            for c in cur_df[cur_df.family_position == "child"].array.values:
                valid_trios.append((m, fathers[0], c))
    parents = [line.rstrip() for line in open(sample_file, "r")]
    # Applies a set of filters here
    valid_filt_trios = []
    for m, f, c in tqdm(valid_trios):
        if (
            (m in parents)
            and (f in parents)
            and find_child_data(c, meta_df, raw_data_path)[1]
        ):
            valid_filt_trios.append((m, f, c))

    return valid_filt_trios


total_data = []
if Path("results/natera_inference/valid_trios.triplets.euploid.txt").is_file():
    with open("results/natera_inference/valid_trios.triplets.euploid.txt", "r") as fp:
        for i, line in enumerate(fp):
            [m, f, _] = line.rstrip().split()
            total_data.append(
                f"results/natera_inference/{m}+{f}.est_recomb.tsv"
            )
        total_data = np.unique(total_data).tolist()


# ------- Rules Section ------- #
localrules:
    all,


rule all:
    input:
        "results/natera_inference/valid_trios.triplets.txt",
        "results/natera_inference/valid_trios.triplets.euploid.txt",
        total_data,


rule generate_parent_sample_list:
    """Generate the parental samples list."""
    input:
        vcf=[vcf_dict[c] for c in chroms],
    output:
        "results/natera_inference/parent_samples.txt",
    resources:
        time="0:30:00",
        mem_mb="1G",
    shell:
        "bcftools query -l {input.vcf} | sort | uniq > {output}"


rule obtain_valid_trios:
    """Obtain the valid trios here."""
    input:
        metadata_tsv=metadata_file,
        parent_samples="results/natera_inference/parent_samples.txt",
    output:
        valid_trios="results/natera_inference/valid_trios.txt",
    resources:
        time="0:30:00",
        mem_mb="1G",
    run:
        valid_trios = create_trios(
            meta_df,
            input.parent_samples,
            raw_data_path="/data/rmccoy22/natera_spectrum/data/",
        )
        with open(output.valid_trios, "w") as out:
            for m, f, c in valid_trios:
                out.write(f"{m}\t{f}\t{c}\n")


rule filter_triplet_pairs:
    """Filter to samples where mother + father have >= 3 embryos."""
    input:
        valid_trios="results/natera_inference/valid_trios.txt",
    output:
        valid_triplets="results/natera_inference/valid_trios.triplets.txt",
    shell:
        "awk 'FNR==NR {{a[$1,$2] += 1;next}} a[$1, $2] >= 3 {{print $0}}' {input.valid_trios} {input.valid_trios} > {output.valid_triplets}"


rule filter_putative_euploid_triplets:
    """Filter to triplets where all individuals were called as disomic across all chromosomes."""
    input:
        aneuploidy_calls=aneuploidy_calls,
    output:
        euploid_triplets="results/natera_inference/valid_trios.triplets.euploid.txt",
    script:
        "scripts/filter_euploid.py"


def define_triplets(mother_id, father_id, trio_file='results/natera_inference/valid_trios.triplets.euploid.txt', base_path='/home/abiddan1/scratch16/natera_aneuploidy/analysis/aneuploidy/results/natera_inference'):
    trio_df = pd.read_csv(trio_file, sep="\t")
    filt_df = trio_df[(trio_df.mother == mother_id) & (trio_df.father == father_id)]
    res = []
    for c in filt_df.child.values:
        res.append(f'{base_path}/{mother_id}+{father_id}/{c}.bafs.pkl.gz')
    return res

rule estimate_recombination_euploid_trio:
    """Estimate crossover events in euploid trio datasets."""
    input:
        euploid_triplets="results/natera_inference/valid_trios.triplets.euploid.txt",
        baf_pkl = lambda wildcards: define_triplets(mother_id = wildcards.mother, father_id=wildcards.father)
    output:
        est_recomb="results/natera_inference/{mother}+{father}.est_recomb.tsv"
    params:
        chroms=chroms
    resources:
        time="1:00:00",
        mem_mb="5G",
    script:
        "scripts/sibling_hmm.py"
