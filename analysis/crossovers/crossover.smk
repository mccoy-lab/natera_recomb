#!python3

import numpy as np
import pandas as pd

import pickle, gzip
from tqdm import tqdm
from pathlib import Path
from io import StringIO

# ---- Parameters for inference in Natera Data ---- #
metadata_file = "../../data/spectrum_metadata_merged.csv"
aneuploidy_calls = "/data/rmccoy22/natera_spectrum/karyohmm_outputs/compiled_output/natera_embryos.karyohmm_v18.bph_sph_trisomy.full_annotation.112023.filter_bad_trios.tsv.gz"

# Create the VCF data dictionary for each chromosome ...
vcf_dict = {}
chroms = [f"chr{i}" for i in range(1, 23)]
for i, c in enumerate(range(1, 23)):
    vcf_dict[
        chroms[i]
    ] = f"/data/rmccoy22/natera_spectrum/genotypes/opticall_parents_100423/genotypes/eagle_phased_hg38/natera_parents.b38.chr{c}.vcf.gz"

# Read in the aggregate metadata file
meta_df = pd.read_csv(metadata_file)


def find_child_data(
    child_id, meta_df=meta_df, raw_data_path="/data/rmccoy22/natera_spectrum/data/"
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
    assert "family_position" in meta_df.columns
    assert "casefile_id" in meta_df.columns
    assert "array" in meta_df.columns
    grouped_df = (
        meta_df.groupby(["casefile_id", "family_position"])["array"]
        .agg(lambda x: list(x))
        .reset_index()
    )
    valid_trios = []
    for case in tqdm(np.unique(grouped_df.casefile_id)):
        cur_df = grouped_df[grouped_df.casefile_id == case]
        for m in cur_df[cur_df.family_position == "mother"].array.values[0]:
            for f in cur_df[cur_df.family_position == "father"].array.values[0]:
                for c in cur_df[cur_df.family_position == "child"].array.values[0]:
                    valid_trios.append((case, m, f, c))

    valid_df = pd.DataFrame(
        valid_trios, columns=["casefile_id", "mother", "father", "child"]
    )
    parents = [line.rstrip() for line in open(sample_file, "r")]
    valid_df["parents_in"] = valid_df.mother.isin(parents) & valid_df.father.isin(
        parents
    )
    valid_df["child_found"] = [
        find_child_data(c)[1] for c in tqdm(valid_df.child.values)
    ]
    valid_filt_df = valid_df[
        valid_df.parents_in & valid_df.child_found
    ].drop_duplicates()[["mother", "father", "child"]]
    valid_filt_trios = [
        (m, f, c)
        for (m, f, c) in zip(
            valid_filt_df.mother.values,
            valid_filt_df.father.values,
            valid_filt_df.child.values,
        )
    ]
    return valid_filt_trios


# Run inference for all valid triplets (restricting to euploid chromosomes)
est_params = False
total_params = []
total_data = []
if Path("results/natera_inference/valid_trios.triplets.txt").is_file():
    with open("results/natera_inference/valid_trios.triplets.txt", "r") as fp:
        for i, line in enumerate(fp):
            [m, f, _] = line.rstrip().split()
            total_data.append(f"results/natera_inference/{m}+{f}.est_recomb.tsv")
            if est_params:
                total_params.append(f"results/natera_inference/{m}+{f}.est_params.tsv")
        total_params = np.unique(total_params).tolist()
        total_data = np.unique(total_data).tolist()


# ------- Rules Section ------- #
localrules:
    all,


rule all:
    input:
        "results/natera_inference/valid_trios.triplets.txt",
        "results/natera_inference/valid_trios.triplets.euploid.txt",
        total_data,
        total_params,


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
            out.write("mother\tfather\tchild\n")
            for m, f, c in valid_trios:
                out.write(f"{m}\t{f}\t{c}\n")


rule filter_triplet_pairs:
    """Filter to samples where mother + father have >= 3 embryos."""
    input:
        valid_trios="results/natera_inference/valid_trios.txt",
    output:
        valid_triplets="results/natera_inference/valid_trios.triplets.txt",
    shell:
        "awk 'FNR==NR {{a[$1,$2] += 1;next}} a[$1, $2] >= 3 {{print $0}}' {input.valid_trios} {input.valid_trios} | awk 'BEGIN{{print \"mother\tfather\tchild\"}}1' > {output.valid_triplets}"


rule filter_putative_euploid_triplets:
    """Filter to triplets where all individuals were called as disomic across all chromosomes."""
    input:
        aneuploidy_calls=aneuploidy_calls,
    output:
        euploid_triplets="results/natera_inference/valid_trios.triplets.euploid.txt",
    params:
        ppThresh=0.95,
    script:
        "scripts/filter_euploid.py"


def define_triplets(
    mother_id,
    father_id,
    trio_file="results/natera_inference/valid_trios.triplets.txt",
    base_path="/home/abiddan1/scratch16/natera_aneuploidy/analysis/aneuploidy/results/november_inference",
):
    trio_df = pd.read_csv(trio_file, sep="\t")
    filt_df = trio_df[(trio_df.mother == mother_id) & (trio_df.father == father_id)]
    res = []
    for c in filt_df.child.values:
        res.append(f"{base_path}/{mother_id}+{father_id}/{c}.bafs.pkl.gz")
    return res


rule est_params_euploid_chrom_trio:
    """Estimate crossover events for euploid chromosomes in trio datasets."""
    input:
        triplets="results/natera_inference/valid_trios.triplets.txt",
        baf_pkl=lambda wildcards: define_triplets(
            mother_id=wildcards.mother, father_id=wildcards.father
        ),
        aneuploidy_calls=aneuploidy_calls,
    output:
        est_params="results/natera_inference/{mother}+{father}.est_params.tsv",
    params:
        chroms=chroms,
        ppThresh=0.95,
    resources:
        time="3:00:00",
        mem_mb="5G",
    script:
        "scripts/est_params_sibling_euploid.py"


rule est_crossover_euploid_chrom_trio:
    """Estimate crossover events for euploid chromosomes in trio datasets."""
    input:
        triplets="results/natera_inference/valid_trios.triplets.txt",
        baf_pkl=lambda wildcards: define_triplets(
            mother_id=wildcards.mother, father_id=wildcards.father
        ),
        aneuploidy_calls=aneuploidy_calls,
    output:
        est_recomb="results/natera_inference/{mother}+{father}.est_recomb.tsv",
        recomb_paths="results/natera_inference/{mother}+{father}.recomb_paths.pkl.gz",
    params:
        chroms=chroms,
        ppThresh=0.95,
    resources:
        time="3:00:00",
        mem_mb="10G",
    script:
        "scripts/sibling_hmm.py"
