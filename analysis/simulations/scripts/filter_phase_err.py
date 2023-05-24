import numpy as np


def filt_phase_err(breakpts, prop=0.5):
    """Filter the phase errors out of the dataset.

    Args:
            breakpts: list (np.array)
            prop: (float) - proportion of samples to be called a phase error.

    NOTE: this should be run separately
    """
    assert len(breakpts) > 1
    assert (prop <= 1.0) and (prop >= 0.0)
    n = len(breakpts)
    # Concatenate the breakpts and obtain the unique counts
    concat_breakpts = np.hstack(breakpts)
    uniq_elements, counts = np.unique(concat_breakpts, return_counts=True)
    breakpt_cnt_dict = {x: c / n for (x, c) in zip(uniq_elements, counts)}
    # Create a set of filtered breakpoints
    filt_breakpts = []
    for i in range(n):
        cur_x = np.sort([c for c in breakpts[i] if breakpt_cnt_dict[c] <= prop])
        filt_breakpts.append(cur_x)
    return filt_breakpts


if __name__ == "__main__":
    nsibs = snakemake.params["nsibs"]
    prop = snakemake.params["prop"]
    data = np.load(snakemake.input["infer_hmm_co"])
    # Obtain maternal and paternal inferred breakpts
    maternal_breakpts = []
    paternal_breakpts = []
    for i in range(nsibs):
        maternal_breakpts.append(data[f"maternal_rec{i}"].tolist())
        paternal_breakpts.append(data[f"paternal_rec{i}"].tolist())
    # Filter the breakpts
    maternal_breakpts_filt = filt_phase_err(maternal_breakpts, prop=prop)
    paternal_breakpts_filt = filt_phase_err(paternal_breakpts, prop=prop)
    res_dict = {}
    for i in range(nsibs):
        res_dict[f"mat_rec_filt_{i}"] = maternal_breakpts_filt[i]
        res_dict[f"pat_rec_filt_{i}"] = paternal_breakpts_filt[i]
    np.savez_compressed(snakemake.output["filt_co"], **res_dict)
