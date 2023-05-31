import gzip as gz
import pickle
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from karyohmm import QuadHMM

def load_baf_data(baf_pkls):
    """Load in the multiple BAF datasets."""
    family_data = {}
    for fp in baf_pkls:
        embryo_name = Path(fp).stem.split('.')[0]
        with gz.open(fp, 'rb') as f:
            data = pickle.load(f)
            family_data[embryo_name] = data
    return family_data

def prepare_paired_data(embryo_id1='10013440016_R06C01', embryo_id2='10013440016_R04C01', embryo_id3='10013440016_R05C01', chrom='chr21', data_dict=None):
    """Create the filtered dataset for evaluating crossovers using the QuadHMM results."""
    data_embryo1 = data_dict[embryo_id1][chrom]
    data_embryo2 = data_dict[embryo_id2][chrom]
    data_embryo3 = data_dict[embryo_id3][chrom]
    if (data_embryo1['pos'].size != data_embryo2['pos'].size) or ((data_embryo2['pos'].size != data_embryo3['pos'].size)):
        pos1 = data_embryo1['pos']
        pos2 = data_embryo2['pos']
        pos3 = data_embryo3['pos']
        idx2 = np.isin(pos2, pos1) & np.isin(pos2, pos3)
        idx1 = np.isin(pos1, pos2) & np.isin(pos1, pos3)
        idx3 = np.isin(pos3, pos1) & np.isin(pos3, pos2)
        baf1 = data_embryo1['baf_embryo'][idx1]
        baf2 = data_embryo2['baf_embryo'][idx2]
        baf3 = data_embryo3['baf_embryo'][idx3]
        mat_haps = data_embryo1['mat_haps'][:,idx1]
        pat_haps = data_embryo1['pat_haps'][:,idx1]
        assert baf1.size == baf2.size
        assert baf2.size == baf3.size
        # Return the maternal haplotypes, paternal haplotypes, baf
        return mat_haps, pat_haps, baf1, baf2, baf3, pos1[idx1]
    else:
        pos = data_embryo1['pos']
        baf1 = data_embryo1['baf_embryo']
        baf2 = data_embryo2['baf_embryo']
        baf3 = data_embryo3['baf_embryo']
        mat_haps = data_embryo1['mat_haps']
        pat_haps = data_embryo1['pat_haps']
        # Return the maternal haplotypes, paternal haplotypes, baf
        assert baf1.size == baf2.size
        assert baf2.size == baf3.size
        return mat_haps, pat_haps, baf1, baf2, baf3, pos

def find_nearest_het(idx, pos, haps):
    """Find the nearest heterozygotes to the estimated crossover position."""
    assert idx >= 0 and idx <= haps.shape[1]
    assert pos.size == haps.shape[1]
    geno = haps[0,:] + haps[1,:]
    het_idx = np.where(geno == 1)[0]
    if idx < np.min(het_idx):
        left_boundary = np.nan
    else:
        left_boundary = pos[het_idx[het_idx <= idx][-1]]
    if idx > np.max(het_idx):
        right_boundary = np.nan
    else:
        right_boundary = pos[het_idx[het_idx >= idx][0]]
    return left_boundary, pos[idx], right_boundary

if __name__ == "__main__":
    # Read in the input data and params ...
    hmm = QuadHMM()
    family_data = load_baf_data(snakemake.input['baf_pkl'])
    names = [k for k in family_data.keys()]
    nsibs = len(names)
    lines = []
    for c in tqdm(snakemake.params["chroms"]):
        for i in range(nsibs):
            j = (i + 1) % nsibs
            j2 = (i + 2) % nsibs
            mat_haps, pat_haps, baf0, baf1, baf2, pos = prepare_paired_data(embryo_id1=names[i], embryo_id2=names[j], embryo_id3=names[j2], chrom=c, data_dict=family_data)
            path_01, _, _,_ = hmm.viterbi_algorithm(bafs=[baf0, baf1], mat_haps=mat_haps, pat_haps=pat_haps, r=1e-18)
            refined_path_01 = hmm.restrict_path(path_01)
            path_02, _, _,_ = hmm.viterbi_algorithm(bafs=[baf0, baf2], mat_haps=mat_haps, pat_haps=pat_haps, r=1e-18)
            refined_path_02 = hmm.restrict_path(path_02)
            mat_rec, pat_rec = hmm.isolate_recomb(refined_path_01, refined_path_02)
            for m in mat_rec:
                left, rec_pos, right = find_nearest_het(m[0], pos, mat_haps)                
                lines.append(f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{names[i]}\t{c}\tmaternal\t{left}\t{rec_pos}\t{right}')
            for p in pat_rec:
                left, rec_pos, right = find_nearest_het(p[0], pos, pat_haps)                
                lines.append(f'{snakemake.wildcards["mother"]}\t{snakemake.wildcards["father"]}\t{names[i]}\t{c}\tpaternal\t{left}\t{rec_pos}\t{right}')
    with open(snakemake.output['est_recomb'], 'w') as out:
        out.write('mother\tfather\tchild\tchrom\tcrossover_sex\tmin_pos\tavg_pos\tmax_pos\n')
        for line in lines:
            out.write(line)