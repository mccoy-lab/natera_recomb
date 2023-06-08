#!python3

import numpy as np 
from scipy import stats
import pandas as pd

def mean_var_co_per_genome(df):
    """Compute the average number of crossovers per-chromosome for an individual."""
    assert 'mother' in df.columns
    assert 'father' in df.columns
    assert 'child' in df.columns
    assert 'crossover_sex' in df.columns
    data = []
    for m in np.unique(df.mother):
        values = df[(df.mother == m) & (df.crossover_sex == 'maternal')].groupby('child').count()['mother'].values
        data.append([m,m, np.mean(values), np.var(values)])
    for p in np.unique(df.father):
        values = df[(df.father == p) & (df.crossover_sex == 'paternal')].groupby('child').count()['mother'].values
        data.append([p,p, np.mean(values), np.var(values)])
    out_df = pd.DataFrame(data)
    out_df.columns = ['FID', 'IID', 'MeanCO', 'VarCO']
    return out_df

def random_pheno(df, seed=42):
    """Create a random phenotype as a test for main analysis."""
    assert 'mother' in df.columns
    assert 'father' in df.columns
    assert 'child' in df.columns
    assert 'crossover_sex' in df.columns
    assert seed > 0
    np.random.seed(seed)
    data = []
    for m in np.unique(df.mother):
        x = np.random.normal()
        data.append([m,m, x])
    for p in np.unique(df.father):
        x = np.random.normal()
        data.append([p,p, x])
    out_df = pd.DataFrame(data)
    out_df.columns = ['FID', 'IID', 'RandPheno']
    return out_df

    
if __name__ == '__main__':
    """Create several phenotypes for analysis of recombination."""
    co_df = pd.read_csv(snakemake.input['co_data'], sep="\t")
    mean_R_df = mean_var_co_per_genome(co_df)
    rand_df = random_pheno(co_df)
    merged_df = mean_R_df.merge(rand_df)
    if snakemake.params['plink_format']:
        merged_df.rename(columns={'FID':"#FID"})
    merged_df.to_csv(snakemake.output['co_data'], sep="\t", index=None)
