import numpy as np
from scipy.stats import norm, rankdata
from sklearn.preprocessing import quantile_transform


def quantile_normalize(x):
    """Apply quantile normalization to a phenotype."""
    assert x.size > 0
    x_norm = quantile_transform(
        x.reshape(-1, 1), n_quantiles=10000, output_distribution="normal", subsample=1e6
    )
    return x_norm


def inverse_rank_transform(x):
    """Inverse-Rank Transformation of a non-normal phenotype."""
    x_rank = rankdata(x, method="average", nan_policy="omit")
    # using Rankit convention to process the rank https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2921808/
    rank_scaled = (x_rank - 0.5) / np.nanmax(x_rank)  # scale ranks
    x_norm = norm.ppf(rank_scaled)
    return x_norm
