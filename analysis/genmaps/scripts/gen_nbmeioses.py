import numpy as np
import pandas as pd
from tqdm import tqdm


def isolate_regions(min_pos, max_pos, nsplit=3, use_raw=False):
    """Isolate the key positions."""
    raw_pos = np.unique(np.sort(min_pos.tolist() + max_pos.tolist())).astype(int)
    split_pos = []
    for i, j in zip(raw_pos[:-1], raw_pos[1:]):
        split_pos.append(np.linspace(i, j, nsplit, dtype=int))
    pos = np.unique(np.sort(np.hstack(split_pos))).astype(int)
    if use_raw:
        starts = raw_pos[:-1]
        ends = raw_pos[1:]
    else:
        starts = pos[:-1]
        ends = pos[1:]
    region_df = pd.DataFrame({"begin": starts, "ends": ends})
    return region_df


if __name__ == "__main__":
    co_df = pd.read_csv(snakemake.input["co_map"], sep="\t")
    sex_spec_co_df = co_df[co_df.crossover_sex == snakemake.params["sex"]]
    nmeioses = np.unique(sex_spec_co_df.child.values).size

    # Estimate the events files per-chromosome ...
    # NOTE: this should be based on SNP-locations I think
    cur_df = sex_spec_co_df[sex_spec_co_df.chrom == snakemake.wildcards["chrom"]][
        ["min_pos", "max_pos"]
    ].dropna()
    # We could use the SNP
    region_chrom_df = isolate_regions(
        cur_df.min_pos.values,
        cur_df.max_pos.values,
        nsplit=snakemake.params["nsplit"],
        use_raw=snakemake.params["use_raw"],
    )
    region_chrom_df["nbmeioses"] = nmeioses
    region_chrom_df.to_csv(
        snakemake.output["nbmeioses"], index=None, header=None, sep="\t"
    )
