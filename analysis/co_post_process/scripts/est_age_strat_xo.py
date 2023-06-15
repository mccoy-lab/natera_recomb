import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from tqdm import tqdm

# Import the xoi inference package here
xoi = importr("xoi")


def create_age_tranches(samples, ages, bins=10):
    """Split age of samples by some amount."""
    assert samples.size == ages.size
    assert samples.ndim == ages.ndim
    assert samples.ndim == 1
    nquant_space = np.linspace(0, 1, num=bins)
    age_tranches = []
    sample_tranches = []
    for i in range(1, bins):
        q1, q2 = nquant_space[i - 1], nquant_space[i]
        age_q1, age_q2 = np.quantile(ages, q1), np.quantile(ages, q2)
        print(age_q1, age_q2)
        age_tranches.append((age_q1, age_q2))
        idx = np.where((ages >= age_q1) & (ages < age_q2))[0]
        cur_samples = samples[idx]
        sample_tranches.append(cur_samples.tolist())
    return age_tranches, sample_tranches


def create_xo_data(co_df, samples, sex="maternal"):
    """Create crossover dataset for inference of xo-interference."""
    if sex == "maternal":
        cur_df = co_df[np.isin(co_df.mother, samples) & (co_df.crossover_sex == sex)]
    elif sex == "paternal":
        cur_df = co_df[np.isin(co_df.father, samples) & (co_df.crossover_sex == sex)]
    else:
        raise ValueError("sex must be either maternal or paternal.")
    xo_data_df = (
        cur_df.groupby(["child", "chrom"])[["min_pos_cM", "avg_pos_cM", "max_pos_cM"]]
        .agg(
            {
                "min_pos_cM": lambda x: list(x),
                "avg_pos_cM": lambda x: list(x),
                "max_pos_cM": lambda x: list(x),
            }
        )
        .reset_index()
    )
    return xo_data_df


def bootstrap_infer_xo(xo_data_df, chrom="chr1", chrom_len=267.77, nboots=50, seed=42):
    """Estimating parameters with bootstrap replicates to estimate std-errors."""
    assert nboots > 0
    assert seed > 0
    np.random.seed(seed)
    # NOTE: we don't conduct sampling from the window of cM positions
    # NOTE: we also limit values that are beyond the interpolation range
    xo_list = xo_data_df[xo_data_df.chrom == chrom].avg_pos_cM.values.tolist()
    xo_list = [np.unique(x).tolist() for x in xo_list]
    npts = len(xo_list)
    try:
        mle_est = xoi.fitStahl(xo_list, chrlen=chrom_len, verbose=False)
        nu_hat = mle_est[0]
        p_hat = mle_est[1]
    except:
        nu_hat = np.nan
        p_hat = np.nan
    nu_boots = np.zeros(nboots)
    p_boots = np.zeros(nboots)
    for i in tqdm(range(nboots)):
        ix = np.random.choice(np.arange(npts), size=npts)
        xo_boot = [xo_list[j] for j in ix]
        try:
            boot_est = xoi.fitStahl(xo_boot, chrlen=chrom_len, verbose=False)
            print(boot_est)
            nu_boots[i] = boot_est[0]
            p_boots[i] = boot_est[1]
        except:
            nu_boots[i] = np.nan
            p_boots[i] = np.nan
    # NOTE: this is the standard deviation (not necessarily the std.err)
    nu_std = np.nanstd(nu_boots)
    p_std = np.nanstd(p_boots)
    return nu_hat, nu_std, p_hat, p_std


if __name__ == "__main__":
    # Read in the meta-data & crossover information
    chroms = [f"chr{x}" for x in range(1, 23)]
    meta_df = pd.read_csv(snakemake.input["metadata"])
    crossover_df = pd.read_csv(snakemake.input["co_map_interp"], sep="\t")
    recmap_df = pd.read_csv(snakemake.input["recmap"], comment="#", sep="\t")
    recmap_df.columns = ["chrom", "begin", "end", "cMperMb", "cM"]
    chrom_len = {}
    for c in chroms:
        # NOTE: we had to add some buffer for the length here
        chrom_len[c] = float(recmap_df[recmap_df.chrom == c].cM.values.max() + 1.0)
    # 1. Estimate the age-stratified interference parameters for maternal
    unique_mothers = np.unique(crossover_df.mother.values)
    mother_df = meta_df[np.isin(meta_df.array, unique_mothers)][
        ["array", "patient_age"]
    ].dropna()
    age_tranches, sample_tranches = create_age_tranches(
        samples=mother_df.array.values,
        ages=mother_df.patient_age.values,
        bins=snakemake.params["nbins"],
    )
    mat_df = []
    for i in range(len(sample_tranches)):
        (min_age, max_age) = age_tranches[i]
        cur_samples = sample_tranches[i]
        xo_data_df = create_xo_data(crossover_df, cur_samples, sex="maternal")
        nu_hat, nu_std, p_hat, p_std = bootstrap_infer_xo(
            xo_data_df,
            chrom=snakemake.wildcards["chrom"],
            chrom_len=chrom_len[snakemake.wildcards["chrom"]],
            nboots=snakemake.params["nboots"],
            seed=snakemake.params["seed"],
        )
        mat_df.append(
            [
                "maternal",
                min_age,
                max_age,
                len(cur_samples),
                c,
                nu_hat,
                nu_std,
                p_hat,
                p_std,
            ]
        )
    mat_df = pd.DataFrame(mat_df)
    mat_df.columns = [
        "sex",
        "min_age",
        "max_age",
        "n",
        "chrom",
        "nu_hat",
        "nu_std",
        "p_hat",
        "p_std",
    ]
    # 2. Estimate the age-stratified interference parameters for paternal crossovers
    unique_fathers = np.unique(crossover_df.father.values)
    father_df = meta_df[np.isin(meta_df.array, unique_fathers)][
        ["array", "partner_age"]
    ].dropna()
    age_tranches, sample_tranches = create_age_tranches(
        father_df.array.values,
        father_df.partner_age.values,
        bins=snakemake.params["nbins"],
    )
    pat_df = []
    for i in range(len(sample_tranches)):
        (min_age, max_age) = age_tranches[i]
        cur_samples = sample_tranches[i]
        xo_data_df = create_xo_data(crossover_df, cur_samples, sex="paternal")
        nu_hat, nu_std, p_hat, p_std = bootstrap_infer_xo(
            xo_data_df,
            chrom=snakemake.wildcards["chrom"],
            chrom_len=chrom_len[snakemake.wildcards["chrom"]],
            nboots=snakemake.params["nboots"],
            seed=snakemake.params["seed"],
        )
        pat_df.append(
            [
                "paternal",
                min_age,
                max_age,
                len(cur_samples),
                c,
                nu_hat,
                nu_std,
                p_hat,
                p_std,
            ]
        )
    pat_df = pd.DataFrame(pat_df)
    pat_df.columns = [
        "sex",
        "min_age",
        "max_age",
        "n",
        "chrom",
        "nu_hat",
        "nu_std",
        "p_hat",
        "p_std",
    ]
    # 3. Concatenate both dataframes and see how this ends up going ...
    tot_df = pd.concat([mat_df, pat_df])
    tot_df.to_csv(snakemake.output["age_sex_interference"], sep="\t", index=None)
