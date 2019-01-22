"""
Function definitions for receptor-state activity, response, 
and gain in CS decoding scheme for olfaction.

Created by Nirag Kadakia at 23:30 07-31-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
from stats import Kk_dist_Gaussian_activity
from scipy.stats import gamma
from scipy.interpolate import interp1d
import warnings
import sys


def linear_gain(Ss0, Kk1, Kk2, eps, comp=True, num_sites=1):
	"""
	Set linearized binding and activation gain. `comp' sets whether the 
	binding is competitive or non-competitive. `num_sites' sets the 
	number of independent binding sites per receptor.
	"""
	
	dAadSs0 = sp.zeros(Kk1.shape)
	Mm = Kk1.shape[0]
	Nn = Kk1.shape[1]

	if comp == True:
	
		Kk1_sum = sp.dot(Kk1**-1.0, Ss0)
		Kk2_sum = sp.dot(Kk2**-1.0, Ss0)
		A0 = 1./(1. + sp.exp(eps)*(1 + Kk1_sum)**num_sites/(1 + Kk2_sum)**num_sites)
		
		for iM in range(Mm):
			WL_term = num_sites*((1./Kk1[iM,:])/(sp.ones(Nn) + Kk1_sum[iM]) - 
							(1./Kk2[iM,:])/(sp.ones(Nn) + Kk2_sum[iM]))
			dAadSs0[iM,:] = -A0[iM]*(sp.ones(Nn) - A0[iM])*WL_term

	else:
		print ('Non-competitive not coded yet')
		quit()
			
	return dAadSs0
	
def receptor_activity(Ss, Kk1, Kk2, eps, comp=True, num_sites=1):
	"""
	Steady state activity with binding and activation 
	Kk2 is the activated disassociation constants (K2)
	Kk1 is the inactivated disassociation constants (K1)
	K1 = Kk1 >> Ss+Ss0 >> Kk2 = K2. 
	Tranpose in the sums allows for an array of stimuli at once; 
	Ss array would have shape (Nn, number of stimuli). `comp' sets 
	whether the binding is competitive or non-competitive. `num_sites' 
	sets the number of indepdent binding sites per receptor.
	"""
	
	if comp == True:
		Kk1_sum = sp.dot(Kk1**-1.0, Ss)
		Kk2_sum = sp.dot(Kk2**-1.0, Ss)
		Aa = (1. + sp.exp(eps.T)*(1 + Kk1_sum.T)**num_sites/
			 (1 + Kk2_sum.T)**num_sites)**-1.0
	else:
		print ('Non-competitive not coded yet')
		quit()
		
	return Aa.T

def temporal_kernel(vec, memory_vec, integration_Tt, kernel_params):
	"""
	Apply temporal kernel to current values of activity or gain levels. 
	This function discretely integrates the convolution of the 
	temporal kernel with a vector of activities held in a finite
	length of memory. To account for the fact that the kernel has
	features at a finer timescale than the integration time, the
	activities are interpolated in between these times to convolve
	on a finer timescale. integration_Tt is the time vector of the 
	full integration of the estimation.
	"""
	
	kernel_T, kernel_dt, kernel_tau_1, kernel_tau_2, kernel_shape_1, \
		kernel_shape_2, kernel_alpha, kernel_scale, = kernel_params

	# Length of memory vector based on signal sampling rate
	signal_dt = integration_Tt[1] - integration_Tt[0]
	signal_Tt = sp.linspace(0, kernel_T, int(kernel_T/signal_dt) + 1)
	memory_vec_len = len(signal_Tt)
	
	# Finer mesh for kernel integration, utilizing kernel_dt
	kernel_Tt = sp.arange(0, kernel_T, kernel_dt)
	
	# Assign memory vector holding past activity levels; replace t = 0 
	# value with current vector value; roll rest to previous times
	if memory_vec is None:
		memory_vec = sp.zeros(vec.shape + (memory_vec_len,))
	memory_vec = sp.roll(memory_vec, 1)
	memory_vec[...,0] = vec
		
	# Interpolating function to get finer mesh for kernel integration
	interp_f = interp1d(signal_Tt, memory_vec)
	
	# Get kernel and Yy, Yy0 at points on scale of kernel_dt
	vec_interped = interp_f(kernel_Tt)
	kernel = kernel_scale*(gamma.pdf(kernel_Tt, kernel_shape_1, 
				scale=kernel_tau_1) - kernel_alpha*gamma.pdf(kernel_Tt, 
				kernel_shape_2, scale=kernel_tau_2))
	
	# Apply the filter
	vec = sp.sum(vec_interped*kernel*kernel_dt, axis=-1)
	
	return vec, memory_vec
	
def free_energy(Ss, Kk1, Kk2, adapted_A0, comp=True, num_sites=1):
	"""
	Adapted steady state free energy for given stimulus level, 
	disassociation constants, and adapted steady state activity level.
	Tranpose in the sums allows for an array of stimuli at once; 
	Ss array would have shape (Nn, number of stimuli), and then eps
	will have shape (Mm, number of stimuli).  `comp' sets whether the 
	binding is competitive or non-competitive. `num_sites' sets the 
	number of binding sites.
	"""
	
	if comp == True:
		Kk1_sum = sp.dot(Kk1**-1.0, Ss)
		Kk2_sum = sp.dot(Kk2**-1.0, Ss)
		epsilon = sp.log((1.- adapted_A0)/adapted_A0*(1. + Kk2_sum.T)**num_sites/
					(1. + Kk1_sum.T)**num_sites)
	else:
		print ('Non-competitive not coded yet')
		quit()
		
	return epsilon.T
		
def Kk2_samples(shape, receptor_activity_mus, receptor_activity_sigmas, 
				Ss0, eps, seed):
	"""
	Generate K_d matrices, assuming known statistics of tuning curves for 
	individual receptors. 
	"""

	Mm, Nn = shape
	Kk2 = sp.zeros(shape)
	
	assert Mm == len(receptor_activity_mus), \
			"Mean receptor activity vector dimension != measurement "\
			"dimension %s" % Mm
	assert Mm == len(receptor_activity_sigmas), \
			"St dev receptor activity vector dimension != measurement "\
			"dimension %s" % Mm
	
	warnings.filterwarnings('error')
	sp.random.seed(seed)
	
	print ("Generating Kk2 matrix rows from Gaussian tuning curves...")
	for iM in range(Mm):
		print(('\nRow %s..' % iM))
		sample_lower_bnd  = -5000
		sample_upper_bnd = 5000
		slack = 0.4
		bounds_too_lax = True
		while (bounds_too_lax == True) and (sample_upper_bnd > 1e-8):
			try:
				args_dict = dict(activity_mu=receptor_activity_mus[iM], 
								activity_sigma=receptor_activity_sigmas[iM], 
								Ss0=Ss0, eps=eps, size=Nn)
				Kk2_rv_object = Kk_dist_Gaussian_activity(a=sample_lower_bnd,
														b=sample_upper_bnd)
				Kk2[iM, :]  = (Kk2_rv_object.rvs(**args_dict))
				assert sp.all(Kk2[iM, :] != sample_lower_bnd), \
											"Lower bound hit"
				assert sp.all(Kk2[iM, :] != sample_upper_bnd), \
											"Upper bound hit"
				assert sp.all(Kk2[iM, :] != 0), "zero"
				assert sp.unique(Kk2[iM, :]).size > int(Nn*0.9), \
											 "Many values equal"
				bounds_too_lax = False				
			except Warning as e:
				sample_lower_bnd = sample_lower_bnd*slack
				sample_upper_bnd = sample_upper_bnd*slack
				print(('No convergence; bounds --> %s, %s..   ' \
						% (sample_lower_bnd, sample_upper_bnd)))
				pass
			except AssertionError as e:
				sample_lower_bnd = sample_lower_bnd*slack
				sample_upper_bnd = sample_upper_bnd*slack
				print(('%s; bounds --> %s, %s..   ' \
						% (e, sample_lower_bnd, sample_upper_bnd)))
				pass
		print ('..OK')
	print ("\nKk2 matrix successfully generated\n")
	
	return Kk2

def Kk2_eval_normal_activity(shape, receptor_activity_mus, 
								receptor_activity_sigmas, Ss0, eps, seed):
	"""
	Generate K_d matrices, assuming known statistics of tuning curves for 
	individual receptors, by sampling a, then changing variabels to Kk2.
	"""
	
	Mm, Nn = shape
	Kk2 = sp.zeros(shape)
	
	assert Mm == len(receptor_activity_mus), \
			"Mean receptor activity vector dimension != measurement "\
			"dimension %s" % Mm
	assert Mm == len(receptor_activity_sigmas), \
			"St dev receptor activity vector dimension != measurement "\
			"dimension %s" % Mm
	
	C = sp.exp(-eps)*Ss0 
	
	sp.random.seed(seed)
	for iM in range(Mm):
		activity = sp.random.normal(receptor_activity_mus[iM], 
									receptor_activity_sigmas[iM], Nn)
		Kk2[iM, :] = (1./activity - 1.)*C
	
	return Kk2

def Kk2_eval_uniform_activity(shape, params, Ss0, eps, seed):
	"""
	Generate K_d matrices, assuming known statistics of tuning curves for 
	individual receptors, by sampling a, then changing variabels to Kk2.
	"""
	
	lo = params[0]
	hi = params[1]
	
	C = sp.exp(-eps)*Ss0 

	sp.random.seed(seed)
	matrix_activity = sp.random.uniform(lo, hi, shape)
	Kk2 = (1./matrix_activity - 1.)*C
	
	return Kk2

def Kk2_eval_exponential_activity(shape, receptor_activity_mus, 
									Ss0, eps, seed):
	"""
	Generate K_d matrices, assuming known statistics of tuning curves for 
	individual receptors, by sampling a, then changing variables to Kk2.
	"""
	
	Mm, Nn = shape
	Kk2 = sp.zeros(shape)
	
	assert Mm == len(receptor_activity_mus), \
			"Mean receptor activity vector dimension != measurement "\
			"dimension %s" % Mm
	
	C = sp.exp(-eps)*Ss0 
	
	sp.random.seed(seed)
	for iM in range(Mm):
		activity = sp.random.exponential(receptor_activity_mus[iM], Nn)
		Kk2[iM, :] = (1./activity - 1.)*C
	
	return Kk2
	
def inhibitory_normalization(Yy, C, D, eta, R):
	"""
	Add inhibitory divisive normalization.
	"""
	
	total_act = sp.sum(Yy)
	Mm = len(Yy)
	
	return 1.*R*(Yy**eta)/(Yy**eta + 1.*C/Mm*total_act + D)
	
def inhibitory_normalization_linear_gain(Yy0, Rr, C, D, eta, R):
	"""
	Chain rule propagated via divisive normalization to the gain matrix.
	"""
	
	Mm = len(Yy0)
	df_da = sp.zeros((Mm, Mm))
	total_act = sp.sum(Yy0)
	
	for iM in range(Mm):	
		den = (Yy0[iM]**eta + 1.*C/Mm*total_act + D)**2.0
		df_da[iM, :] = R*(-(Yy0[iM]**eta)*C/Mm)/den
		df_da[iM, iM] += R*(eta*Yy0[iM]**(eta - 1.0)*(1.*C/Mm*total_act + D))/den
	
	df_da_dot_Rr = sp.dot(df_da, Rr)
	
	return df_da_dot_Rr
	
	
	
	