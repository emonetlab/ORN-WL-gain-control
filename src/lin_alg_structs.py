"""
Functions for creating random and non-random linear algebraic structures.

Created by Nirag Kadakia at 23:30 07-31-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import scipy as sp
from scipy.stats import powerlaw


def random_matrix(matrix_shape, params, sample_type='normal', seed=0):
	"""
	Generate random matrix with given distribution
	"""
	
	if sample_type == 'normal':
		sp.random.seed(seed)
		mean, sigma = params[:2]
		if sigma != 0.:
			return sp.random.normal(mean, sigma, matrix_shape)
		else:
			return mean*sp.ones(matrix_shape)
	
	elif sample_type == "rank2_row_gaussian":
		sp.random.seed(seed)
		means, sigmas = params[:2]
		
		assert len(matrix_shape) == 2, \
				"rank2_row_gaussian method needs a 2x2 matrix"
		nRows, nCols = matrix_shape
		assert len(means) == nRows, "rank2_row_gaussian needs " \
										"mu vector of proper length"
		assert len(sigmas) == nRows, "rank2_row_gaussian needs " \
										"sigma vector of proper length"
		out_matrix = sp.zeros(matrix_shape)
		
		for iRow in range(nRows):
			out_matrix[iRow, :] = sp.random.normal(means[iRow], sigmas[iRow], 
													nCols)
		return out_matrix
	
	elif sample_type == 'uniform':
		sp.random.seed(seed)
		lo, hi = params[:2]
		return sp.random.uniform(lo, hi, matrix_shape)
	
	elif sample_type == "rank2_row_uniform":
		sp.random.seed(seed)
		bounds_lo, bounds_hi = params[:2]
		
		assert len(matrix_shape) == 2, \
				"rank2_row_uniform method needs a 2x2 matrix"
		nRows, nCols = matrix_shape
		assert len(bounds_lo) == nRows, "rank2_row_uniform needs " \
										"bounds_lo vector of proper length"
		assert len(bounds_hi) == nRows, "rank2_row_uniform needs " \
										"bounds_hi vector of proper length"
		out_matrix = sp.zeros(matrix_shape)
		
		for iRow in range(nRows):
			out_matrix[iRow, :] = sp.random.uniform(bounds_lo[iRow], 
									bounds_hi[iRow], nCols)
		return out_matrix
	
	elif sample_type == "rank2_row_power":
		sp.random.seed(seed)
		bounds_lo, bounds_hi, power_exp = params[:3]
		
		assert len(matrix_shape) == 2, \
				"rank2_row_power method needs a 2x2 matrix"
		nRows, nCols = matrix_shape
		assert len(bounds_lo) == nRows, "rank2_row_gaussian needs " \
										"bounds_lo vector of proper length"
		assert len(bounds_hi) == nRows, "rank2_row_gaussian needs " \
										"bounds_hi vector of proper length"
		out_matrix = sp.zeros(matrix_shape)
		
		for iRow in range(nRows):
			out_matrix[iRow, :] = powerlaw.rvs(power_exp, loc=bounds_lo[iRow], 
									scale=bounds_hi[iRow], size=nCols)	
		return out_matrix
	
	elif sample_type == 'power':
		sp.random.seed(seed)
		lo, hi, power_exp = params[:3]
		return powerlaw.rvs(power_exp, loc=lo, scale=hi, size=matrix_shape)	
	
	elif sample_type == "gaussian_mixture":
		mean1, sigma1, mean2, sigma2, prob_1 = params[:5]
		assert prob_1 <= 1., "Gaussian mixture needs p < 1" 
		
		sp.random.seed(seed)
		mixture_idxs = sp.random.binomial(1, prob_1, matrix_shape)
		it = sp.nditer(mixture_idxs, flags=['multi_index'])
		out_vec = sp.zeros(matrix_shape)
		
		while not it.finished:
			if mixture_idxs[it.multi_index] == 1: 
				out_vec[it.multi_index] = sp.random.normal(mean1, sigma1)
			else:
				out_vec[it.multi_index] = sp.random.normal(mean2, sigma2)
			it.iternext()
		
		return out_vec
	
	else:
		print ('No proper matrix sample_type!')
		exit()

def sparse_vector(nDims, params, sample_type='normal', seed=0):
	"""
	Set sparse stimulus with given statistics
	"""
	
	Nn, Kk = nDims
	Ss = sp.zeros(Nn)
	
	sp.random.seed(seed)
	
	for iK in range(Kk): 
		if sample_type == "normal":
			mu, sigma = params
			if sigma != 0:
				Ss[iK] = sp.random.normal(mu, sigma)
			else:
				Ss[iK] = mu
		elif sample_type == "uniform":
			lo, hi = params
			Ss[iK] = sp.random.uniform(lo,hi)
	
	sp.random.shuffle(Ss)
	idxs = sp.nonzero(Ss)
	
	return Ss, idxs
	
def sparse_vector_bkgrnd(nDims, idxs, params, sample_type='normal', seed=0):
	"""
	Set sparse stimulus background on nonzero components
	of a sparse vector, componenents in list 'idxs'
	"""

	Nn, Kk = nDims
	Ss = sp.zeros(Nn)
	Ss_noisy = sp.zeros(Nn)
	
	sp.random.seed(seed)

	for iK in idxs: 
		if sample_type == "normal":
			mu, sigma = params
			Ss[iK] = mu
			if sigma != 0:
				Ss_noisy[iK] += sp.random.normal(mu, sigma)
			else:
				Ss_noisy[iK] += mu
		elif sample_type == "uniform":
			lo, hi = params
			Ss[iK] = lo + (hi - lo)/2.
			Ss_noisy[iK] += sp.random.uniform(lo, hi)
	
	return Ss, Ss_noisy

def manual_sparse_vector(nDims, idxs, params, seed=0):
	"""
	Set sparse stimulus background manually, components
	listed in manual_dSs
	"""
	
	dSs = sp.zeros(nDims)
	mu, sigma = params
	
	sp.random.seed(seed)
	
	for idx, dSs_idx in enumerate(idxs):
		if sigma != 0:
			dSs[int(dSs_idx)] = sp.random.normal(mu, sigma) 
		else:
			dSs[int(dSs_idx)] = mu
	
	return dSs