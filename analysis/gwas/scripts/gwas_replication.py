import polars as pl

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
    # 3. Creating a joining of the two results
    joint_gwas_results_df = gwas_results_df.join(
        haldorsson_df, left_on=["ID"], right_on=["ID1"], how="left"
    ).join(haldorsson_df, left_on=["ID"], join_nulls=True, right_on=["ID2"], how="left")
    joint_gwas_results_df.write_csv(snakemake.output["final_sumstats"], separator="\t")
