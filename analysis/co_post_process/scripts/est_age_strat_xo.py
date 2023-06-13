import numpy as np 
import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Import the xoi inference here ... 
xoi = importr('xoi')


def create_age_tranches(samples, ages, bins=10):
	"""Split age of samples by some amount."""
	assert samples.size = ages.size
	nquant_space = np.linspace(0, 1, num=bins)
	age_tranches = []
	samples = []
	for i in range(1, bins):
		q1,q2 = nquant_space[i-1], nquant_space[i]
		age_q1, age_q2 = np.quantile(ages, q1), np.quantile(ages, q2)	
		age_tranches.append((age_q1, age_q2))
		cur_samples = samples[(ages >= age_q1) & (ages < age_q2)]
		samples.append(cur_samples)
	return age_tranches, samples


def create_xo_data(co_df, samples, sex="maternal"):
	"""create crossover dataset for inference."""
	if sex == "maternal":
		cur_df = co_df[np.isin(co_df.mother, samples) & co_df.crossover_sex == sex]
	elif sex == "paternal":
		cur_df = co_df[np.isin(co_df.father, samples) & co_df.crossover_sex == sex]
	else:
		raise ValueError("sex must be either maternal or paternal.")


if __name__ == "__main__":
	pass

