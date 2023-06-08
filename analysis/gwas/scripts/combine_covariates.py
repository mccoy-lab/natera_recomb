import pandas as pd
import numpy as np 

def generate_parent_meta(meta_df):
    for a in ["family_position", "array", "patient_age", "partner_age", "year"]:
        assert a in meta_df.columns
    parent_meta_df = meta_df[(meta_df.family_position == 'mother') | (meta_df.family_position == 'father')]
    parent_meta_df = parent_meta_df.groupby(['array', 'family_position']).agg(np.mean).reset_index()
    parent_meta_df = parent_meta_df.assign(sex = [0 if x == 'mother' else 1 for x in parent_meta_df['family_position']], age = [a if x == 'mother' else b for (x,a,b) in zip(parent_meta_df['family_position'], parent_meta_df['patient_age'], parent_meta_df['partner_age'])])
    parent_meta_out_df = parent_meta_df[['array', 'array', 'sex', 'age', 'year']]
    parent_meta_out_df.columns = ['IID', 'FID', 'Sex', 'Age', 'Year']
    return parent_meta_out_df

if __name__ == "__main__":
    """Creating covariates for analyses of traits."""
    meta_df = pd.read_csv(snakemake.input['metadata'])
    parent_meta_df = generate_parent_meta(meta_df)
    pc_df = pd.read_csv(snakemake.input['evecs'], sep="\t")
    pc_df.rename(columns={"#FID":"FID"}, inplace=True)
    final_meta_df = parent_meta_df.merge(pc_df)
    if snakemake.params['plink_format']:
        final_meta_df.rename(columns={"FID":"#FID"}, inplace=True)
    final_meta_df.to_csv(snakemake.output['covars'], sep="\t", index=None)
