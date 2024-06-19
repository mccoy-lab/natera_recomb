"""MLE estimates of escape proportion and intensity parameters per-sample."""

# Actually install the R library for estimating crossover interference
import subprocess

import numpy as np
import pandas as pd

subprocess.run(
    "Rscript -e 'install.packages('xoi', repos='http://cran.us.r-project.org')'"
)
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from scipy.optimize import minimize
from tqdm import tqdm

xoi = importr("xoi")

def est_eta_nu_params(xolocs, chrom_lens):
    """Estimate MLE parameters for crossover-interference in the Housworth-Stahl model for a single individual."""

    def loglikStahl(nu, p):
        loglik = 0.0
        for c in chroms:
            loglik += xoi.stahlLoglik(xolocs[c], chrlen=chrom_lens[c], nu=nu, p=p)[0]
        return -loglik

    # Actually performing the inference here to estimate the parameters
    opt_res = minimize(
        loglikStahl,
        x0=[4, 0.1],
        method="L-BFGS-B",
        bounds=[(1, 10), (0.01, 0.99)],
        tol=1e-3,
        options={"disp": False},
    )
    nu_est = opt_res.x[0]
    p_est = opt_res.x[1]
    return nu_est, p_est


if __name__ == "__main__":
    pass