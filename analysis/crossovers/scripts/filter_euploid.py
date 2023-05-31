import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    # Reading in the prior whole-chromosome aneuploidy calls
    aneuploidy_df = pd.read_csv(snakemake.input['aneuploidy_calls'], sep="\t")
    assert 'bf_max_cat' in aneuploidy_df.columns
    assert 'mother' in aneuploidy_df.columns
    assert 'father' in aneuploidy_df.columns
    assert 'child' in aneuploidy_df.columns
    trios_data_euploid = aneuploidy_df.groupby(['mother', 'father', 'child'])['bf_max_cat'].agg(lambda x: np.sum(x == '2') == 22).reset_index()
    true_euploid_data = trios_data_euploid[trios_data_euploid.bf_max_cat].groupby(['mother', 'father', 'child']).count().reset_index()
    df = true_euploid_data.pivot_table(index = ['mother', 'father'], aggfunc ='size').reset_index()
    parent_dict_count = {f'{m}+{f}': z for (m,f,z) in zip(df.mother.values,df.father.values, df[0].values)}
    x = np.zeros(true_euploid_data.shape[0])
    for i, (m,f) in enumerate(zip(true_euploid_data.mother.values, true_euploid_data.father.values)):
        if f'{m}+{f}' in parent_dict_count:
            x[i] = parent_dict_count[f'{m}+{f}']
    true_euploid_data['n_embryos'] = x
    euploid_trios_df = true_euploid_data[true_euploid_data.n_embryos >= 3][['mother', 'father', 'child']]
    euploid_trios_df.to_csv(snakemake.output['euploid_triplets'], sep="\t", index=None)
                
