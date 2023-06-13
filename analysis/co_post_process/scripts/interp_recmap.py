import pandas as pd
import numpy as np
from tqdm import tqdm 
from scipy.interpolate import interp1d

if __name__ == '__main__':
	crossover_df = pd.read_csv(snakemake.input["co_map"], sep="\t")
	recmap_df = pd.read_csv(snakemake.input["recmap"], comment="#", sep="\t")
	recmap_df.columns = ['chrom', 'begin', 'end', 'cMperMb', 'cM']
	# Create the list and catalogue of chromosomes for interpolation from previous genetic maps
	unique_chroms = np.unique(crossover_df.chrom.values)
	interp_dict = {}
	for c in tqdm(unique_chroms):
		cur_df = recmap_df[recmap_df.chrom == c]
		interp_dict[c] = interp1d(cur_df.begin.values, cur_df.cM.values, fill_value="extrapolate")

	# Creating the interpolated points
	min_pos_cM = np.zeros(crossover_df.shape[0])
	max_pos_cM = np.zeros(crossover_df.shape[0])
	avg_pos_cM = np.zeros(crossover_df.shape[0])
	for i, c in tqdm(enumerate(unique_chroms)):
		idx = np.where(crossover_df.chrom == c)[0]
		cur_df = crossover_df[crossover_df.chrom == c]
		cur_min_pos_cM = interp_dict[c](cur_df.min_pos.values)
		cur_max_pos_cM = interp_dict[c](cur_df.max_pos.values)
		cur_avg_pos_cM = interp_dict[c](cur_df.avg_pos.values)
		min_pos_cM[idx] = cur_min_pos_cM
		avg_pos_cM[idx] = cur_avg_pos_cM
		max_pos_cM[idx] = cur_max_pos_cM
	crossover_df['min_pos_cM'] = min_pos_cM
	crossover_df['avg_pos_cM'] = avg_pos_cM
	crossover_df['max_pos_cM'] = max_pos_cM
	crossover_df.to_csv(snakemake.output["co_map_interp"], sep="\t", index=None)
