import numpy as np
import pandas as pd


def generate_co_meta(co_df):
    """Method that creates some metrics for how noisy each embryo may be."""
    for a in ["mother", "father", "avg_pi0", "avg_sigma"]:
        assert a in co_df.columns
    mother_df = (
        co_df.groupby("mother")[["avg_pi0", "avg_sigma"]].agg("mean").reset_index()
    )
    mother_df = mother_df[["mother", "mother", "avg_pi0", "avg_sigma"]]
    mother_df.columns = ["FID", "IID", "AVGPI0", "AVGSIGMA"]
    father_df = (
        co_df.groupby("father")[["avg_pi0", "avg_sigma"]].agg("mean").reset_index()
    )
    father_df = father_df[["father", "father", "avg_pi0", "avg_sigma"]]
    father_df.columns = ["FID", "IID", "AVGPI0", "AVGSIGMA"]
    res_df = pd.concat([mother_df, father_df])
    return res_df


def generate_parent_meta(meta_df):
    """Generate parental covariates for the embryos."""
    for a in ["family_position", "array", "patient_age", "partner_age", "year"]:
        assert a in meta_df.columns
    parent_meta_df = meta_df[
        (meta_df.family_position == "mother") | (meta_df.family_position == "father")
    ]
    parent_meta_df = (
        parent_meta_df.groupby(["array", "family_position"]).agg(np.mean).reset_index()
    )
    parent_meta_df = parent_meta_df.assign(
        sex=[0 if x == "mother" else 1 for x in parent_meta_df["family_position"]],
        age=[
            a if x == "mother" else b
            for (x, a, b) in zip(
                parent_meta_df["family_position"],
                parent_meta_df["patient_age"],
                parent_meta_df["partner_age"],
            )
        ],
    )
    parent_meta_out_df = parent_meta_df[["array", "array", "sex", "age"]]
    parent_meta_out_df.columns = ["FID", "IID", "Sex", "Age"]
    return parent_meta_out_df


if __name__ == "__main__":
    """Creating covariates for analyses of traits."""
    co_df = pd.read_csv(snakemake.input["co_data"], sep="\t")
    meta_df = pd.read_csv(snakemake.input["metadata"])
    co_meta_df = generate_co_meta(co_df)
    parent_meta_df = generate_parent_meta(meta_df)
    parent_meta_df = parent_meta_df.merge(co_meta_df)
    pc_df = pd.read_csv(snakemake.input["evecs"], sep="\t")
    pc_df.rename(columns={"#FID": "FID"}, inplace=True)
    final_meta_df = parent_meta_df.merge(pc_df)
    if snakemake.params["plink_format"]:
        final_meta_df.rename(columns={"FID": "#FID"}, inplace=True)
    # final_meta_df.dropna(inplace=True)
    final_meta_df.to_csv(snakemake.output["covars"], na_rep="NA", sep="\t", index=None)
