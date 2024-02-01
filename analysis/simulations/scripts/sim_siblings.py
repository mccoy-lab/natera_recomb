#!python3

import numpy as np
from karyohmm import PGTSim

if __name__ == "__main__":
    pgt_sim = PGTSim()
    if snakemake.params["sfs"] != "None":
        # Estimate the simulated allele frequency parameters
        afs = np.loadtxt(snakemake.params["sfs"])
    else:
        afs = None

    # Set the seed as unique for this seed, phase_error, nsib combination ...
    seed = (
        snakemake.params["seed"]
        + int(snakemake.params["phase_err"] * 1e3)
        + int(snakemake.params["nsib"] * 100)
    )
    # Run the full simulation using the defined helper function
    # NOTE: This now should have an expected number of one crossover per chromosome per parent
    r = 1.0 / snakemake.params["m"]
    table_data = pgt_sim.sibling_euploid_sim(
        afs=afs,
        m=snakemake.params["m"],
        length=1e7,
        nsibs=snakemake.params["nsib"],
        rec_prob=r,
        std_dev=snakemake.params["sigma"],
        mix_prop=snakemake.params["pi0"],
        switch_err_rate=snakemake.params["phase_err"],
        seed=seed,
    )
    np.savez_compressed(snakemake.output["sim"], **table_data)
