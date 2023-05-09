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
	raise NotImplementedError("This function will be implemented soon!")


if __name__ == '__main__':
	data = np.load(input.)
	pass