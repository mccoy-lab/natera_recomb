#!python3

import numpy as np
import polars as pl

import pickle, gzip
from tqdm import tqdm
from pathlib import Path
from io import StringIO

# ---- Parameters for inference in Natera Data ---- #
metadata_file = "../../data/spectrum_metadata_merged.csv"
aneuploidy_calls = "/data/rmccoy22/natera_spectrum/karyohmm_outputs/compiled_output/natera_embryos.karyohmm_v30a.bph_sph_trisomy.full_annotation.031624.tsv.gz"

total_trisomy_data = []
if Path("results/natera_inference_trisomy/valid_trisomies.filt.tsv").is_file():
    with open("results/natera_inference_trisomy/valid_trisomies.filt.tsv", "r") as fp:
        for i, line in enumerate(fp):
            if i > 0:
                [m, f, c, chrom, _, _, _, _] = line.rstrip().split()
                total_trisomy_data.append(
                    f"results/natera_inference_trisomy/{m}+{f}+{c}.{chrom}.est_recomb_trisomy.tsv"
                )


# ------- Rules Section ------- #
localrules:
    all,


rule all:
    input:
        "results/natera_inference_trisomy/valid_trisomies.filt.tsv",
        total_trisomy_data,


def define_baf(
    mother_id,
    father_id,
    child_id,
    trio_file="results/natera_inference/valid_trios.triplets.txt",
    base_path="/home/abiddan1/scratch16/natera_aneuploidy/analysis/aneuploidy/results/natera_inference",
):
    return f"{base_path}/{mother_id}+{father_id}/{child_id}.bafs.pkl.gz"


def define_hmm(
    mother_id,
    father_id,
    child_id,
    base_path="/home/abiddan1/scratch16/natera_aneuploidy/analysis/aneuploidy/results/natera_inference",
):
    return f"{base_path}/{mother_id}+{father_id}/{child_id}.hmm_model.pkl.gz"


# --------- Identifying crossovers in trisomies via BPH vs SPH transitions -------- #
rule isolate_trisomies:
    """Isolate potential trisomies that we should test using BPH vs. SPH transitions."""
    input:
        aneuploidy_calls=aneuploidy_calls,
    output:
        trisomy_calls="results/natera_inference_trisomy/valid_trisomies.tsv",
    params:
        ppThresh=0.90,
    run:
        aneu_df = pl.read_csv(
            input.aneuploidy_calls, separator="\t", null_values=["NA"]
        )
        trisomy_df = aneu_df.filter(~pl.col("embryo_noise_3sd")).filter(
            (pl.col("3m") > params["ppThresh"]) | (pl.col("3p") > params["ppThresh"])
        )
        trisomy_df[
            [
                "mother",
                "father",
                "child",
                "chrom",
                "bf_max_cat",
                "pi0_baf",
                "sigma_baf",
                "post_max",
            ]
        ].write_csv(output.trisomy_calls, null_value="NA", separator="\t")


rule filter_trisomy_calls:
    input:
        trisomy_calls="results/natera_inference_trisomy/valid_trisomies.tsv",
    output:
        filt_trisomy_calls="results/natera_inference_trisomy/valid_trisomies.filt.tsv",
    run:
        import polars as pl
        from pathlib import Path
        from tqdm import tqdm

        trisomy_df = pl.read_csv(
            input.trisomy_calls, separator="\t", null_values=["NA"]
        )
        unique_trisomy_df = trisomy_df.unique(["mother", "father", "child"])[
            "mother", "father", "child"
        ]
        filt = np.zeros(unique_trisomy_df.shape[0], dtype=bool)
        i = 0
        for mother, father, child in tqdm(
            zip(
                unique_trisomy_df["mother"].to_numpy(),
                unique_trisomy_df["father"].to_numpy(),
                unique_trisomy_df["child"].to_numpy(),
            )
        ):
            hmm_fp = define_hmm(mother_id=mother, father_id=father, child_id=child)
            baf_fp = define_baf(mother_id=mother, father_id=father, child_id=child)
            filt[i] = Path(hmm_fp).exists() & Path(baf_fp).exists()
            i += 1
        unique_trisomy_df = unique_trisomy_df.with_columns(
            pl.Series(filt).alias("file_exists")
        )
        trisomy_df = trisomy_df.join(
            unique_trisomy_df, how="inner", on=["mother", "father", "child"]
        ).filter(pl.col("file_exists"))
        trisomy_df[
            [
                "mother",
                "father",
                "child",
                "chrom",
                "bf_max_cat",
                "pi0_baf",
                "sigma_baf",
                "post_max",
            ]
        ].write_csv(output.filt_trisomy_calls, null_value="NA", separator="\t")


rule est_crossover_trisomic_chrom_trio:
    """Estimate crossovers using trisomy-specific path tracing."""
    input:
        trisomy_calls="results/natera_inference_trisomy/valid_trisomies.filt.tsv",
        hmm_pkl=lambda wildcards: define_hmm(
            mother_id=wildcards.mother,
            father_id=wildcards.father,
            child_id=wildcards.child,
        ),
        baf_pkl=lambda wildcards: define_baf(
            mother_id=wildcards.mother,
            father_id=wildcards.father,
            child_id=wildcards.child,
        ),
    output:
        trisomy_recomb="results/natera_inference_trisomy/{mother}+{father}+{child}.{chrom}.est_recomb_trisomy.tsv",
    params:
        penalty=100,
        min_size=500,
    script:
        "scripts/sibling_co_trisomy.py"
