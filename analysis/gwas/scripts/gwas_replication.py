import numpy as np
import polars as pl
from tqdm import tqdm


def variant_replicates(cpra_id, ids_dict):
    """Return if the variant replicates."""
    return cpra_id in ids_dict


def fraction_replicates(cpra_ids, ids_dict):
    """Obtain the fraction of variants that are replicated from a set."""
    return np.mean([variant_replicates(x, ids_dict) for x in cpra_ids])


def gene_replicates(gene_name, gene_dict):
    """Determine if the gene replicates."""
    return gene_name in gene_dict


if __name__ == "__main__":
    # 1. Read in the original summary statistic dataset!
    gwas_results_df = pl.read_csv(snakemake.input["sumstats"], separator="\t")
    gwas_results_df = gwas_results_df.with_columns(
        pl.col("Gencode")
        .str.extract('gene_name\s"[A-Z|0-9|\-|a-z|.]+"', 0)
        .str.replace_all('"', "")
        .str.split(" ")
        .list.get(1)
        .alias("Gene"),
        (pl.col("P") < 0.05 / 7215119).alias("Bonferroni"),
    ).sort(pl.col("P"))
    # 2. Read in the replication dataset!
    haldorsson_df = pl.read_csv(snakemake.input["replication"])
    haldorsson_df.columns = [f"{x}_Haldorsson19" for x in haldorsson_df.columns]
    haldorsson_df = haldorsson_df.with_columns(
        pl.concat_str(
            [
                pl.col("Chrom_Haldorsson19"),
                pl.col("Pos_Haldorsson19"),
                pl.col("Amaj_Haldorsson19"),
                pl.col("Amin_Haldorsson19"),
            ],
            separator=":",
        ).alias("ID1"),
        pl.concat_str(
            [
                pl.col("Chrom_Haldorsson19"),
                pl.col("Pos_Haldorsson19"),
                pl.col("Amin_Haldorsson19"),
                pl.col("Amaj_Haldorsson19"),
            ],
            separator=":",
        ).alias("ID2"),
    )
    # 3. Creating new vectors for per-variant, cluster, and gene, replication
    haldorsson_ids = np.hstack(
        [haldorsson_df["ID1"].to_numpy(), haldorsson_df["ID2"].to_numpy()]
    )
    haldorsson_ids_dict = {v: 1 for v in haldorsson_ids}
    # 3a. Perform variant-level replication
    cpras = gwas_results_df["ID"].to_numpy()
    variant_clusters = gwas_results_df["SP2"].to_numpy()
    var_replicates = np.zeros(cpras.size, dtype=bool)
    var_cluster_frac_replicates = np.zeros(cpras.size)
    for i, v in tqdm(enumerate(cpras)):
        var_replicates[i] = variant_replicates(v, haldorsson_ids_dict)
        var_cluster_frac_replicates[i] = fraction_replicates(
            variant_clusters[i].split(","), haldorsson_ids_dict
        )
    # 3b. Perform gene-level replication of effects
    df = haldorsson_df.group_by("gene_Haldorsson19").agg(
        pl.col("Phenotype_Haldorsson19").unique()
    )
    genes = gwas_results_df["Gene"].to_numpy()
    gene_dict = {x for x in df["gene_Haldorsson19"].unique().to_numpy()}
    gene_phenotype_dict = {
        x: y
        for (x, y) in zip(
            df["gene_Haldorsson19"].to_numpy(),
            df["Phenotype_Haldorsson19"].to_list(),
        )
    }
    gene_replicates_agg = np.zeros(cpras.size)
    gene_rep_phenotypes_agg = []
    for i, g in tqdm(enumerate(genes)):
        gene_replicates_agg[i] = gene_replicates(g, gene_dict)
        if gene_replicates(g, gene_dict):
            gene_rep_phenotypes_agg.append(",".join(gene_phenotype_dict[g]))
        else:
            gene_rep_phenotypes_agg.append("")
    gwas_results_df = gwas_results_df.with_columns(
        pl.Series(name="Variant_replicates", values=var_replicates),
        pl.Series(name="Frac_cluster_replicates", values=var_cluster_frac_replicates),
        pl.Series(name="Gene_replicates", values=gene_replicates_agg),
        pl.Series(name="Gene_replicates_phenotypes", values=gene_rep_phenotypes_agg),
    )
    gwas_results_df.write_csv(snakemake.output["sumstats_replication"], separator="\t")
