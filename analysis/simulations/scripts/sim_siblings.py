#!python3

import numpy as np
from scipy.stats import beta, binom, norm, rv_histogram, truncnorm, uniform

# These are the different classes of aneuploidy that we can putatively simulate from
sim_ploidy_values = ["0", "1m", "1p", "2", "3m", "3p"]

# Setup dictionaries for LRR estimation
lrr_mu = {0: -3.527211, 1: np.log2(0.5), 2: np.log2(1.0), 3: np.log2(1.5)}
lrr_sd = {0: 1.329152, 1: 0.284338, 2: 0.159645, 3: 0.209089}


def est_beta_params(xs, eps=1e-3):
    """Estimated parameters of a beta distribution."""
    xs_filt = xs[(xs > eps) & (xs < 1.0 - eps)]
    mu_, var_ = np.mean(xs_filt), np.var(xs_filt)
    a = ((1 - mu_) / var_ - 1 / mu_) * (mu_**2)
    b = a * (1 / mu_ - 1)
    assert (a > 0) and (b > 0)
    return a, b


def draw_parental_genotypes(afs=None, m=100, seed=42):
    """Draw parental genotypes from a beta distribution.

    Args:
        afs (`np.`): alpha parameter of a beta distribution.
        m (`int`): number of variants to simulate.
        seed (`int`): random number seed.
    Output:
        maternal_haps (`np.array`): maternal haplotypes
        paternal_haps (`np.array`): paternal haplotypes

    """
    assert m > 0
    assert seed > 0
    np.random.seed(seed)
    if afs is None:
        # Draw from a uniform distribution ...
        ps = beta.rvs(1.0, 1.0, size=m)
    else:
        # This is the case where we actually have an AFS ...
        assert afs.size > 10
        rv = rv_histogram(
            np.histogram(afs, bins=np.min([100, afs.size / 20]).astype(np.int32))
        )
        ps = rv.rvs(size=m)
    # Simulate diploid parental haplotypes
    mat_h1 = binom.rvs(1, ps)
    mat_h2 = binom.rvs(1, ps)
    pat_h1 = binom.rvs(1, ps)
    pat_h2 = binom.rvs(1, ps)
    # NOTE: assuming diploid here ...
    return [np.vstack([mat_h1, mat_h2]), np.vstack([pat_h1, pat_h2])]


def create_switch_errors(mat_haps, pat_haps, err_rate=1e-3, seed=42):
    """Create switch errors to evaluate impact of poor phasing."""
    assert mat_haps.size == pat_haps.size
    assert mat_haps.ndim == 2
    assert mat_haps.ndim == 2
    assert (err_rate > 0) and (err_rate < 1)
    assert seed > 0
    # Create the shuffled maternal haplotypes
    m = mat_haps.shape[1]
    mat_haps_prime = np.zeros(shape=mat_haps.shape, dtype=int)
    pat_haps_prime = np.zeros(shape=mat_haps.shape, dtype=int)
    mi0, mi1 = 0, 1
    pi0, pi1 = 0, 1
    us1 = np.random.uniform(size=m)
    us2 = np.random.uniform(size=m)
    m_switch = np.where(us1 < err_rate)[0]
    p_switch = np.where(us2 < err_rate)[0]
    for i in range(m):
        if us1[i] < err_rate:
            mi0 = 1 - mi0
            mi1 = 1 - mi1
        mat_haps_prime[0, i] = mat_haps[mi0, i]
        mat_haps_prime[1, i] = mat_haps[mi1, i]
        if us2[i] < err_rate:
            pi0 = 1 - pi0
            pi1 = 1 - pi1
        pat_haps_prime[0, i] = pat_haps[pi0, i]
        pat_haps_prime[1, i] = pat_haps[pi1, i]
    # Create the shuffled paternal haplotypes
    return mat_haps_prime, pat_haps_prime, m_switch, p_switch


def sim_haplotype_paths(mat_haps, pat_haps, ploidy=2, rec_prob=1e-2, seed=42):
    """Simulate paths through the maternal and paternal haplotypes."""
    assert ploidy == 2
    assert mat_haps.size == pat_haps.size
    np.random.seed(seed)
    m = mat_haps.shape[1]
    # Simulating the hidden variables ...
    zs_maternal = np.zeros(m, dtype=np.uint16)
    zs_paternal = np.zeros(m, dtype=np.uint16)

    # A typical euploid sample ...
    zs_maternal[0] = binom.rvs(1, 0.5)
    zs_paternal[0] = binom.rvs(1, 0.5)
    for i in range(1, m):
        # We switch with a specific probability
        zs_maternal[i] = (
            1 - zs_maternal[i - 1] if uniform.rvs() <= rec_prob else zs_maternal[i - 1]
        )
        zs_paternal[i] = (
            1 - zs_paternal[i - 1] if uniform.rvs() <= rec_prob else zs_paternal[i - 1]
        )
    mat_real_hap = np.array([mat_haps[i, p] for p, i in enumerate(zs_maternal)])
    pat_real_hap = np.array([pat_haps[i, p] for p, i in enumerate(zs_paternal)])
    aploid = "2"
    return zs_maternal, zs_paternal, mat_real_hap, pat_real_hap, aploid


def sim_b_allele_freq(mat_hap, pat_hap, ploidy=2, std_dev=0.2, mix_prop=0.3, seed=42):
    """Simulate of B-allele frequency."""
    np.random.seed(seed)
    assert (ploidy <= 3) & (ploidy >= 0)
    assert mat_hap.size == pat_hap.size
    true_geno = mat_hap + pat_hap
    baf = np.zeros(true_geno.size)
    for i in range(baf.size):
        if ploidy == 0:
            baf[i] = np.random.uniform()
        else:
            mu_i = true_geno[i] / ploidy
            a, b = (0 - mu_i) / std_dev, (1 - mu_i) / std_dev
            if mu_i == 0:
                baf[i] = (
                    0.0
                    if uniform.rvs() < mix_prop
                    else truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
                )
            elif mu_i == 1:
                baf[i] = (
                    1.0
                    if uniform.rvs() < mix_prop
                    else truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
                )
            else:
                baf[i] = truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
    return true_geno, baf


def sim_logR_ratio(mat_hap, pat_hap, ploidy=2, alpha=1.0, seed=42):
    """Simulate logR-ratio conditional on ploidy.

    Alpha is the degree to which the variance is increased for the LRR.
    """
    assert seed > 0
    assert ploidy in [0, 1, 2, 3]
    assert mat_hap.size == pat_hap.size
    np.random.seed(seed)
    m = mat_hap.size
    lrr = norm.rvs(lrr_mu[ploidy], scale=lrr_sd[ploidy] * alpha, size=m)
    return lrr


def sibling_euploid_sim(
    afs=None,
    ploidy=2,
    m=10000,
    nsibs=5,
    rec_prob=1e-4,
    std_dev=0.2,
    mix_prop=0.3,
    alpha=1.0,
    switch_err_rate=1e-2,
    seed=42,
):
    """Simulate euploid embryos that are siblings."""
    assert ploidy == 2
    assert m > 0
    assert seed > 0
    assert nsibs > 0
    np.random.seed(seed)

    res_table = {}
    mat_haps, pat_haps = draw_parental_genotypes(afs=None, m=m, seed=seed)
    mat_haps_prime, pat_haps_prime, mat_switch, pat_switch = create_switch_errors(
        mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
    )
    res_table["mat_haps_true"] = mat_haps
    res_table["pat_haps_true"] = pat_haps
    res_table["mat_haps_real"] = mat_haps_prime
    res_table["pat_haps_real"] = pat_haps_prime
    res_table["mat_switch"] = mat_switch
    res_table["pat_switch"] = pat_switch
    res_table["nsibs"] = nsibs
    res_table["aploid"] = "2"
    for i in range(nsibs):
        zs_maternal, zs_paternal, mat_hap1, pat_hap1, aploid = sim_haplotype_paths(
            mat_haps,
            pat_haps,
            ploidy=ploidy,
            rec_prob=rec_prob,
            seed=seed + i,
        )
        # print(ploidy, aploid, mat_hap1, pat_hap1)
        geno, baf = sim_b_allele_freq(
            mat_hap1,
            pat_hap1,
            ploidy=ploidy,
            std_dev=std_dev,
            mix_prop=mix_prop,
            seed=seed + i,
        )
        lrr = sim_logR_ratio(
            mat_hap1, pat_hap1, ploidy=ploidy, alpha=alpha, seed=seed + i
        )

        assert geno.size == m
        assert baf.size == m
        res_table[f"baf_embryo{i}"] = baf
        res_table[f"lrr_embryo{i}"] = lrr
        res_table[f"zs_maternal{i}"] = zs_maternal
        res_table[f"zs_paternal{i}"] = zs_paternal
    return res_table


if __name__ == "__main__":
    if snakemake.params["sfs"] != "None":
        # Estimate the simulated allele frequency parameters
        # NOTE: could potentially just do with numpy as well...
        afs = np.loadtxt(snakemake.params["sfs"])
    else:
        afs = None
    # Run the full simulation using the defined helper function
    # NOTE: that r=1e-4 indicates something like a 100 Mb simulator ...
    table_data = sibling_euploid_sim(
        afs=afs,
        m=snakemake.params["m"],
        nsibs=snakemake.params["nsib"],
        rec_prob=1e-4,
        std_dev=snakemake.params["sigma"],
        mix_prop=snakemake.params["pi0"],
        alpha=1.0,
        switch_err_rate=snakemake.params["phase_err"],
        seed=snakemake.params["seed"],
    )
    np.savez_compressed(snakemake.output["sim"], **table_data)
