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
    euploid_triplets = []
    parent_dict = {}
    # Open up to having parental trios here .
    with open(snakemake.input['triplets'], 'r') as fp:
        for line in tqdm(fp):
            [mother, father, child] = line.split()
            test_df = aneuploidy_df[(aneuploidy_df.mother == mother) & (aneuploidy_df.father == father) & (aneuploidy_df.child == child)]
            if np.sum(test_df.bf_max_cat.values == '2') == 22:
                euploid_triplets.append((mother, father, child))
                if f'{mother}+{father}' not in parent_dict:
                    parent_dict[f'{mother}+{father}'] = 1
                else:    
                    parent_dict[f'{mother}+{father}'] += 1
    # Write out the files here...
    with open(snakemake.output['euploid_triplets'], 'w') as out:
        for (mother, father, child) in euploid_triplets:
            if parent_dict[f'{mother}+{father}'] >= 3:
                out.write(f'{mother}\t{father}\t{child}\n')
                
    

    