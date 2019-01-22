"""
Object to encode and decode sparse signals using compressed sensing
but with passing a sparse odor signal through a sensory system 
described by a 4-state receptor system. Off and on states 
are distinguished here, and binding kinetics are assumed fast 
enough to leave near the steady state limit. 

The response matrix is assumed to be a linearization of the full
nonlinear response. This linearization is essentially the matrix of 
inverse disassociation constants. This script tests the decoding 
fidelity for various choices of the  mean value of the inverse K. 

Created by Nirag Kadakia at 23:30 07-31-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import sys
sys.path.append('../src')
from lin_alg_structs import random_matrix, sparse_vector, \
							sparse_vector_bkgrnd, manual_sparse_vector
from kinetics import linear_gain, receptor_activity, free_energy, \
						Kk2_samples, Kk2_eval_normal_activity, \
						Kk2_eval_exponential_activity, \
						Kk2_eval_uniform_activity, inhibitory_normalization, \
						inhibitory_normalization_linear_gain, \
						temporal_kernel
from optimize import decode_CS, decode_nonlinear_CS
from utils import clip_array
from load_data import load_signal_trace_from_file, load_Hallem_firing_rate_data


INT_PARAMS = ['Nn', 'Kk', 'Mm', 'seed_Ss0', 'seed_dSs', 'seed_Kk1', 
				'seed_Kk2', 'seed_receptor_activity', 'Kk_split', 
				'Kk_1', 'Kk_2', 'seed_dSs_2']


class four_state_receptor_CS(object):	
	"""	
	Object for encoding and decoding a four-state receptor model 
	using compressed sensing
	"""

	def __init__(self, **kwargs):
	
		# Initialize needed data structures
		self.Kk1 = None
		self.Kk2 = None
		self.Ss0 = None
		self.dSs = None
		self.Ss = None
		self.Yy0 = None
		self.dYy = None
		self.Yy = None
		self.eps = None
		
		# Set system parameters; Kk_split (or Kk_1/Kk_2) for two-level signals
		self.Nn = 50
		self.Kk = 5
		self.Mm = 20
		self.Kk_split = None
		self.Kk_1 = None
		self.Kk_2 = None

		# Set random seeds
		self.seed_Ss0 = 1
		self.seed_dSs = 1
		self.seed_dSs_2 = None
		self.seed_Kk1 = 1
		self.seed_Kk2 = 1
		self.seed_inhibition = 1
		self.seed_eps = 1
		self.seed_receptor_activity = 1
		self.seed_adapted_activity = 1
		
		# Randomly chosen sparse signal background and fluctuations
		self.mu_Ss0 = 1.
		self.sigma_Ss0 = 0.001
		self.mu_dSs = 0.3
		self.sigma_dSs = 0.1
		self.mu_dSs_2 = None
		self.sigma_dSs_2 = None
		
		# Manual signals; numpy array of nonzero components
		self.manual_dSs_idxs = None
		
		# Binding competitive or not, and how many independent binding
		# sites per receptor?
		self.binding_competitive = True
		self.num_binding_sites = 1
		
		# All K1 and K2 from a single Gaussian distribution
		self.mu_Kk1 = 1e4
		self.sigma_Kk1 = 1e3
		self.mu_Kk2 = 1e-3
		self.sigma_Kk2 = 1e-4
		
		# All K1 and K2 from a mixture of 2 Gaussians
		self.mu_Kk1_1 = 1e4
		self.sigma_Kk1_1 = 1e3
		self.mu_Kk1_2 = 1e4
		self.sigma_Kk1_2 = 1e3
		self.mu_Kk2_1 = 1e-3
		self.sigma_Kk2_1 = 1e-4
		self.mu_Kk2_2 = 1e-3
		self.sigma_Kk2_2 = 1e-4
		self.Kk1_p = 0.5
		self.Kk2_p = 0.5
		
		# All K1 and K2 from a single uniform or power distribution,
		# with uniform priors on bounds for each receptor.
		self.lo_Kk1_hyper_lo = 1e2
		self.lo_Kk1_hyper_hi = 1e2
		self.hi_Kk1_hyper_lo = 1e2
		self.hi_Kk1_hyper_hi = 1e2
		self.lo_Kk2_hyper_lo = 1e-4
		self.lo_Kk2_hyper_hi = 1e-4
		self.hi_Kk2_hyper_lo = 1e-3
		self.hi_Kk2_hyper_hi = 1e-3
		self.power_exp = 0.38
		
		# K1 and K2: each receptor from a distinct Gaussian, 
		# with uniform prior on means and sigmas
		self.mu_Kk1_hyper_lo = 1e4
		self.mu_Kk1_hyper_hi = 1e4
		self.sigma_Kk1_hyper_lo = 1e3
		self.sigma_Kk1_hyper_hi = 1e3
		self.mu_Kk2_hyper_lo = 1e-3
		self.mu_Kk2_hyper_hi = 1e-3
		self.sigma_Kk2_hyper_lo = 1e-4
		self.sigma_Kk2_hyper_hi = 1e-4
		
		# Replace some Kk2 values with special responders. 
		# This is done by getting tuning curves for stimulus level of 
		# manual_Kk_replace_Ss and then replacing the weakest responders
		# (average activity less than manual_Kk_replace_min_act with 
		# uniformly distributed values of 10**uniform[manual_Kk_replace_lo, 
		# manual_Kk_replace_hi]
		self.high_responders = False
		self.manual_Kk_replace_Ss = 25
		self.manual_Kk_replace_min_act = 0.03
		self.manual_Kk_replace_lo = -4.5
		self.manual_Kk_replace_hi = -3
		
		# Divisive normalization constants. 
		# a_DN = R*(a^\eta)/(a^\eta + C/M*sum(a) + D)
		self.inh_C = 1.0
		self.inh_D = 1e-9
		self.inh_eta = 1.5
		self.inh_R = 1.0
		self.divisive_normalization = 0.0
		
		# Fixed activity distributions for adapted individual odorant response, 
		# where each activity is chosen from mixture of 2 Gaussians
		self.activity_p = 0.5
		self.receptor_tuning_mixture_mu_1 = 0.5
		self.receptor_tuning_mixture_mu_2 = 0.5
		self.receptor_tuning_mixture_sigma_1 = 0.1
		self.receptor_tuning_mixture_sigma_2 = 0.1
		
		# Fixed activity distributions for adapted individual odorant response, 
		# where each activity is chosen from a single uniform distribution
		self.uniform_activity_lo = 1e-5
		self.uniform_activity_hi = 1e-4
 		
		# Fixed activity distributions for adapted individual odorant response, 
		# where each activity is chosen from a normal distribution depending on
		# the receptor. The means for each receptor are chosen normally and the 
		# sigmas are chosen uniformly. 
		self.receptor_tuning_mu_hyper_mu = 0.2
		self.receptor_tuning_mu_hyper_sigma = 0
		self.receptor_tuning_sigma_hyper_lo = 0
		self.receptor_tuning_sigma_hyper_hi = 0.3
		
		# Random free energy statistics
		self.mu_eps = 5.0
		self.sigma_eps = 0.0
		self.mu_min_eps = 0.0
		self.sigma_min_eps = 0.0
		self.mu_max_eps = 100.0
		self.sigma_max_eps = 0.0
		
		# Tuned free energy maximum
		self.normal_eps_tuning_prefactor = 0.0
		self.normal_eps_tuning_width = 1.0
		self.WL_scaling = 0.0
		
		# Fix tuning curve statistics for adapted full activity
		self.adapted_activity_mu = 0.5
		self.adapted_activity_sigma = 0.01
		
		# Use measured data and generate epsilons and Kk from there.
		self.measured_eps = None
		
		# Added to receptor activity before decoding
		self.meas_noise = 1e-3
		
		# Firing rate from receptor activity
		# Temporal kernel is a liner combination of Gamma distributions K_1, 
		# K_2, with beta = 1/tau_m: K = kernel_scale*K_1 
		# + kernel_alpha*K_2). kernel_scale should be chosen so that the
		# integral of first lobe is 1. Nonlinearity is a thresholded linear.
		# kernel_T holds length in seconds over which kernel acts;
		# kernel_dt holds step size in seconds of kernel integration. 
		# firing_max is maximum firing rate.
		self.kernel_tau_1 = 0.006
		self.kernel_tau_2 = 0.008
		self.kernel_shape_1 = 2
		self.kernel_shape_2 = 3
		self.kernel_alpha = 0.5
		self.kernel_scale = 1./0.63
		self.kernel_T = 0.1
		self.kernel_dt = 5e-4
		self.NL_threshold = 0.05
		self.NL_scale = 300
		self.firing_max = 300
		
		# Temporal coding variables. temporal_adaptation_type can be 'perfect'
		# or 'imperfect'. In the latter case, the rate is used to adapt in 
		# time. Dual odors can be used by setting signal_trace_file_2, 
		# signal_trace_multiplier_2, and signal_trace_offset_2.
		# if temporal_adaptation_rate_sigma is not None, (set to float), then 
		# the adaptation rate is chosen from a spread, seeded with 
		# temporal_adaptation_rate_seed, with mean temporal_adaptation_rate
		# and deviation temporal_adaptation_rate_sigma. Note that the deviation
		# is implemented as a power, such that beta --> 
		# beta*(10^[-sigma, sigma]), affecting the rate over powers. Next,
		# temporal_adaptation_rate_ordering is either 'random', 
		# 'increasing_Yy', 'increasing_dYy', 'decreasing_Yy', or 
		# 'decreasing_dYy', depending on whether it's randomly chosen, 
		# ordered by activity, or ordered opposite to the receptor activity 
		# ordering. Finally, the perfectly adapted levels are set through
		# adapted_activity_mu and adapted_activity_sigma, listed above.
		# self.filter_dT is the length of time, in seconds, over which the 
		# filter will integrate
		self.temporal_run = False
		self.signal_trace_file = None
		self.signal_trace_multiplier = 1.0
		self.signal_trace_offset = 0
		self.signal_trace_file_2 = None
		self.signal_trace_multiplier_2 = 1.0
		self.signal_trace_offset_2 = 0
		self.set_mu_Ss0_temporal_signal = True
		self.temporal_adaptation_type = 'perfect' 
		self.temporal_adaptation_rate = 1.5
		self.temporal_adaptation_rate_sigma = 0
		self.temporal_adaptation_rate_seed = 1
		self.memory_Yy0 = None
		self.memory_Yy = None
		self.memory_Rr = None
				
		# Overwrite variables with passed arguments	
		for key in kwargs:
		
			assert hasattr(self, '%s' % key), "'%s' is not an attribute of "\
				"four_state_receptor_CS class. Check or add to __init__" % key
		
			if key in INT_PARAMS:
				exec ('self.%s = int(kwargs[key])' % key)
			else:
				exec ('self.%s = kwargs[key]' % key)


				
	######################################################
	########## 		Stimulus functions			##########
	######################################################

	
	
	def set_sparse_signals(self):
		"""
		Set random sparse signals
		"""
	
		# Possibly override manual indices
		if self.manual_dSs_idxs is not None:
			self.set_manual_signals()
			return
		
		# Can override Kk_split with this; manually set compelxity of signal 2
		if (self.Kk_1 is not None) and (self.Kk_2 is not None):
			self.Kk = self.Kk_1 + self.Kk_2
			self.Kk_split = self.Kk_2
			
		params_dSs = [self.mu_dSs, self.sigma_dSs]
		params_Ss0 = [self.mu_Ss0, self.sigma_Ss0]
		self.dSs, self.idxs = sparse_vector([self.Nn, self.Kk], 
												params_dSs,	seed=self.seed_dSs)
		
		# Replace components with conflicting background odor 
		if (self.Kk_split is not None) and (self.Kk_split != 0):
			assert 0 <= self.Kk_split <= self.Kk, \
				"Splitting sparse signal into two levels requires Kk_split" \
				" to be non-negative and less than or equal to Kk."
			assert self.mu_dSs_2 is not None \
				and self.sigma_dSs_2 is not None, \
				"Splitting sparse signal into two levels requires that" \
				" mu_dSs_2 and sigma_dSs_2 are set."

			if self.seed_dSs_2 is None:
				
				# Want both odor 1 and odor 2 to be determined by seed_dSs
				sp.random.seed(self.seed_dSs)
				self.idxs_2 = sp.random.choice(self.idxs[0], self.Kk_split, 
											replace=False)
			else:
				
				# Want odor 1 and 2 to be determined by respective seeds.
				self.dSs, self.idxs = sparse_vector([self.Nn, self.Kk_1], 
												params_dSs,	seed=self.seed_dSs)
				_, self.idxs_2 = sparse_vector([self.Nn, self.Kk_split], 
												params_dSs,	seed=self.seed_dSs_2)
				
				# Re-define the full idxs and the odor 2 idxs subset
				self.idxs = ((list(self.idxs[0]) + list(self.idxs_2[0])), )
				self.idxs_2 = self.idxs_2[0]
				
			for idx_2 in self.idxs_2:
				self.dSs[idx_2] = random_matrix(1, params=[self.mu_dSs_2,
												self.sigma_dSs_2])
		else:
			self.idxs_2 = []
			self.Kk_split = 0
			
		# Ss0 is the ideal (learned) background stimulus without noise
		self.Ss0, self.Ss0_noisy = sparse_vector_bkgrnd([self.Nn, self.Kk], 
														self.idxs, params_Ss0,
														seed=self.seed_Ss0)
		
		self.Ss = self.dSs + self.Ss0_noisy
		
	def set_manual_signals(self):
		"""
		Set manually-selected sparse signals. 
		"""
		
		params_dSs = [self.mu_dSs, self.sigma_dSs]
		params_Ss0 = [self.mu_Ss0, self.sigma_Ss0]
		self.idxs = self.manual_dSs_idxs
		self.dSs = manual_sparse_vector(self.Nn, self.manual_dSs_idxs, 
										params_dSs, seed=self.seed_dSs)
		
		# Ss0 is the ideal (learned) background stimulus without noise
		self.Ss0, self.Ss0_noisy = sparse_vector_bkgrnd([self.Nn, self.Kk], 
														self.manual_dSs_idxs, 
														params_Ss0, 
														seed=self.seed_Ss0)
		
		# The true signal, including background noise
		self.Ss = self.dSs + self.Ss0_noisy
	


	######################################################
	########## 		Energy functions			##########
	######################################################

	
	
	def set_adapted_free_energy(self):
		"""
		Set free energy based on adapted activity activity.
		"""
		
		activity_stats = [self.adapted_activity_mu, self.adapted_activity_sigma]
		adapted_activity = random_matrix([self.Mm], params=activity_stats, 
									seed=self.seed_adapted_activity)
		self.eps = free_energy(self.Ss, self.Kk1, self.Kk2, adapted_activity, 
							   self.binding_competitive, self.num_binding_sites)
		
		# Apply max and min epsilon value to each component
		self.min_eps = random_matrix(self.Mm, params=[self.mu_min_eps, 
									self.sigma_min_eps], seed=self.seed_eps)
		self.max_eps = random_matrix(self.Mm, params=[self.mu_max_eps, 
									self.sigma_max_eps], seed=self.seed_eps)
		self.eps = sp.maximum(self.eps.T, self.min_eps).T
		self.eps = sp.minimum(self.eps.T, self.max_eps).T
				
	def set_normal_free_energy(self):
		"""
		Set free energy as a function of odorant; normal tuning curve.
		"""
		
		self.eps_base = self.mu_eps + self.normal_eps_tuning_prefactor* \
						sp.exp(-(1.*sp.arange(self.Mm))**2.0/(2.0* \
						self.normal_eps_tuning_width)**2.0)
						
		self.eps_base += random_matrix(self.Mm, params=[0, self.sigma_eps], 
										seed=self.seed_eps)
		
		# If dual signal, use the average of the FULL signal nonzero components
		if self.Kk_split == 0:
			self.eps = self.WL_scaling*sp.log(self.mu_Ss0) + self.eps_base 
		else:
			self.eps = self.WL_scaling*sp.log(sp.average(self.Ss\
							[self.Ss != 0])) + self.eps_base
		
		# Apply max and min epsilon value to each component
		self.min_eps = random_matrix(self.Mm, params=[self.mu_min_eps, 
									self.sigma_min_eps], seed=self.seed_eps)
		self.max_eps = random_matrix(self.Mm, params=[self.mu_max_eps, 
									self.sigma_max_eps], seed=self.seed_eps)
		self.eps = sp.maximum(self.eps, self.min_eps)
		self.eps = sp.minimum(self.eps, self.max_eps)
			
		# If an array of signals, replicate for each signal.
		if len(self.Ss.shape) > 1:
			self.eps = sp.tile(self.eps, [self.Ss.shape[1], 1]).T
			
	
	######################################################
	########## 		Binding functions			##########
	######################################################
	
						
	def set_power_Kk(self):
		"""
		Set K1 and K2 martrices where each receptor response is chosen from
		a power law. The clip_vals is a 2-element array that enforces the 
		values to be drawn from a reduced range.		
		"""
		
		Kk1_los = random_matrix([self.Mm], params=[self.lo_Kk1_hyper_lo, 
							self.lo_Kk1_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk1)
		Kk1_his = random_matrix([self.Mm], params=[self.hi_Kk1_hyper_lo, 
							self.hi_Kk1_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk1)
		Kk2_los = random_matrix([self.Mm], params=[self.lo_Kk2_hyper_lo, 
							self.lo_Kk2_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk2)
		Kk2_his = random_matrix([self.Mm], params=[self.hi_Kk2_hyper_lo, 
							self.hi_Kk2_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk2)
		
		self.Kk1 = random_matrix([self.Mm, self.Nn], [Kk1_los, Kk1_his, 
								self.power_exp], sample_type='rank2_row_power',
								seed = self.seed_Kk1)
		self.Kk2 = random_matrix([self.Mm, self.Nn], [Kk2_los, Kk2_his, 
								self.power_exp], sample_type='rank2_row_power', 
								seed = self.seed_Kk2)
		
		# Replace some low responders with specialist responders
		if self.high_responders == True:
			self.manual_Kk_replace()
		
		
	def set_mixture_Kk(self, clip=True):
		"""
		Set K1 and K2 matrices where each receptor response is chosen from 
		a Gaussian mixture with stats mu_Kk1_1, sigma_Kk2_1, 
		mu_Kk1_2, sigma_Kk2_2.
		"""
		
		assert 0 <= self.Kk1_p <= 1., "Kk1 Mixture ratio must be between 0 and 1"
		assert 0 <= self.Kk2_p <= 1., "Kk2 Mixture ratio must be between 0 and 1"
		
		self.Kk1 = sp.zeros((self.Mm, self.Nn))
		self.Kk2 = sp.zeros((self.Mm, self.Nn))
		
		num_comp1 = int(self.Kk1_p*self.Mm)
		num_comp2 = self.Mm - num_comp1
		params_Kk1_1 = [self.mu_Kk1_1, self.sigma_Kk1_1]
		params_Kk1_2 = [self.mu_Kk1_2, self.sigma_Kk1_2]
		self.Kk1[:num_comp1, :] = random_matrix([num_comp1, self.Nn], 
										params_Kk1_1, seed = self.seed_Kk1)
		self.Kk1[num_comp1:, :] = random_matrix([num_comp2, self.Nn], 
										params_Kk1_2, seed = self.seed_Kk1)
		
		num_comp1 = int(self.Kk2_p*self.Mm)
		num_comp2 = self.Mm - num_comp1
		params_Kk2_1 = [self.mu_Kk2_1, self.sigma_Kk2_1]
		params_Kk2_2 = [self.mu_Kk2_2, self.sigma_Kk2_2]
		
		self.Kk2[:num_comp1, :] = random_matrix([num_comp1, self.Nn], 
										params_Kk2_1, seed = self.seed_Kk2)		
		self.Kk2[num_comp1:, :] = random_matrix([num_comp2, self.Nn], 
										params_Kk2_2, seed = self.seed_Kk2)
		
		if clip == True:
			array_dict = clip_array(dict(Kk1 = self.Kk1, Kk2 = self.Kk2))
			self.Kk1 = array_dict['Kk1']
			self.Kk2 = array_dict['Kk2']
												
	def set_normal_Kk(self, clip=True):	
		"""
		Set K1 and K2 where each receptor from a distinct Gaussian with 
		uniform prior on means and sigmas (mu_Kk1_hyper_lo, mu_Kk1_hyper_hi, 
		sigma_Kk1_hyper_lo, sigma_Kk1_hyper_hi, etc.)
		"""
		
		Kk1_mus = random_matrix([self.Mm], params=[self.mu_Kk1_hyper_lo, 
							self.mu_Kk1_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk1)
		Kk1_sigmas = random_matrix([self.Mm], params=[self.sigma_Kk1_hyper_lo, 
							self.sigma_Kk1_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk1)
		Kk2_mus = random_matrix([self.Mm], params=[self.mu_Kk2_hyper_lo, 
							self.mu_Kk2_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk2)
		Kk2_sigmas = random_matrix([self.Mm], params=[self.sigma_Kk2_hyper_lo,
							self.sigma_Kk2_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk2)
		
		self.Kk1 = random_matrix([self.Mm, self.Nn], [Kk1_mus, Kk1_sigmas], 
									sample_type='rank2_row_gaussian', 
									seed = self.seed_Kk1)
		self.Kk2 = random_matrix([self.Mm, self.Nn], [Kk2_mus, Kk2_sigmas],
									sample_type='rank2_row_gaussian', 
									seed = self.seed_Kk2)
		

		if clip == True:
			array_dict = clip_array(dict(Kk1 = self.Kk1, Kk2 = self.Kk2))
			self.Kk1 = array_dict['Kk1']
			self.Kk2 = array_dict['Kk2']
	
	def set_uniform_Kk(self, clip=True):
		"""
		Set K1 and K2 where each receptor from a distinct uniform with 
		uniform prior on the uniform bounds (lo_Kk1_hyper_lo, lo_Kk1_hyper_hi, 
		hi_Kk1_hyper_lo, hi_Kk1_hyper_hi, etc.)
		"""
		
		Kk1_los = random_matrix([self.Mm], params=[self.lo_Kk1_hyper_lo, 
							self.lo_Kk1_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk1)
		Kk1_his = random_matrix([self.Mm], params=[self.hi_Kk1_hyper_lo, 
							self.hi_Kk1_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk1)
		Kk2_los = random_matrix([self.Mm], params=[self.lo_Kk2_hyper_lo, 
							self.lo_Kk2_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk2)
		Kk2_his = random_matrix([self.Mm], params=[self.hi_Kk2_hyper_lo, 
							self.hi_Kk2_hyper_hi], sample_type='uniform',
							seed=self.seed_Kk2)
		
		self.Kk1 = random_matrix([self.Mm, self.Nn], [Kk1_los, Kk1_his], 
								sample_type='rank2_row_uniform', 
								seed = self.seed_Kk1)
		self.Kk2 = random_matrix([self.Mm, self.Nn], [Kk2_los, Kk2_his], 
								sample_type='rank2_row_uniform', 
								seed = self.seed_Kk2)
		
		
		if clip == True:
			array_dict = clip_array(dict(Kk1 = self.Kk1, Kk2 = self.Kk2))
			self.Kk1 = array_dict['Kk1']
			self.Kk2 = array_dict['Kk2']
					
					
	def manual_Kk_replace(self):
		"""
		Manually add high responders to Kk2 matrix
		"""
		
		# First get the epsilon -- needed to calculate activities to find 
		# weak responders.
		self.set_normal_free_energy()
		
		# Get tuning curves
		tuning_curves = sp.zeros((self.Mm, self.Nn))
		for iN in range(self.Nn):
			tuning_Ss = sp.zeros(self.Nn)
			tuning_Ss[iN] = self.manual_Kk_replace_Ss
			tuning_curves[:, iN] = receptor_activity(tuning_Ss, self.Kk1, 
														self.Kk2, self.eps, 
														self.binding_competitive, 
														self.num_binding_sites)

		# Add specialist responses to weak responders
		for iM in range(self.Mm):
			if sp.mean(tuning_curves[iM, :]) < self.manual_Kk_replace_min_act:
				self.Kk2[iM, sp.random.randint(self.Nn)] = \
					10.**sp.random.uniform(self.manual_Kk_replace_lo, 
					self.manual_Kk_replace_hi)

							
						
	######################################################
	########## 		Activity functions			##########
	######################################################


						
	def set_Kk2_normal_activity(self, clip=True, **kwargs):
		"""
		Fixed activity distributions for adapted individual odorant response, 
		where each activity is chosen from a normal distribution depending on
		the receptor. The means for each receptor are chosen normally and the 
		sigmas are chosen uniformly. Kk2 is then derived from these choices,
		while Kk1 is chosen from a single Gaussian distribution. The function
		allows for overriding of mu_eps and mu_Ss0 if system has adapted to 
		another background state.
		"""
		
		matrix_shape = [self.Mm, self.Nn]
		
		params_Kk1 = [self.mu_Kk1, self.sigma_Kk1]
		self.Kk1 = random_matrix(matrix_shape, params_Kk1, seed=self.seed_Kk1)
	
		mu_Ss0 = self.mu_Ss0
		mu_eps = self.mu_eps
		for key in kwargs:
			exec ('%s = kwargs[key]' % key)
		
		mu_stats = [self.receptor_tuning_mu_hyper_mu, 
					self.receptor_tuning_mu_hyper_sigma]
		sigma_stats = [self.receptor_tuning_sigma_hyper_lo, 
						self.receptor_tuning_sigma_hyper_hi]
		activity_mus = random_matrix([self.Mm], params=mu_stats, 
										sample_type='normal',
										seed=self.seed_receptor_activity)
		activity_sigmas = random_matrix([self.Mm], params=sigma_stats, 
										sample_type='uniform',
										seed=self.seed_receptor_activity)
		
		self.Kk2 = Kk2_eval_normal_activity(matrix_shape, activity_mus, 
										activity_sigmas, mu_Ss0, mu_eps, 
										self.seed_Kk2)
		
		if clip == True:
			array_dict = clip_array(dict(Kk1 = self.Kk1, Kk2 = self.Kk2))
			self.Kk1 = array_dict['Kk1']
			self.Kk2 = array_dict['Kk2']
	
	def set_Kk2_uniform_activity(self, clip=True, **kwargs):
		"""
		Fixed activity distributions for adapted individual odorant response, 
		where each activity is chosen from a single uniform distribution.
		"""
		
		matrix_shape = [self.Mm, self.Nn]
		
		params_Kk1 = [self.mu_Kk1, self.sigma_Kk1]
		self.Kk1 = random_matrix(matrix_shape, params_Kk1, seed=self.seed_Kk1)
	
		mu_Ss0 = self.mu_Ss0
		mu_eps = self.mu_eps
		for key in kwargs:
			exec ('%s = kwargs[key]' % key)
		
		params_Kk2 = [self.uniform_activity_lo, self.uniform_activity_hi]
		self.Kk2 = Kk2_eval_uniform_activity(matrix_shape, params_Kk2, 
											mu_Ss0, mu_eps, 
											self.seed_Kk2)
		
		if clip == True:
			array_dict = clip_array(dict(Kk1 = self.Kk1, Kk2 = self.Kk2))
			self.Kk1 = array_dict['Kk1']
			self.Kk2 = array_dict['Kk2']
	
	def set_Kk2_normal_activity_mixture(self, clip=True, **kwargs):
		"""
		Fixed activity distributions for adapted individual odorant response, 
		where each activity is chosen from a Gaussian mixture.
		"""
		
		matrix_shape = [self.Mm, self.Nn]
		
		params_Kk1 = [self.mu_Kk1, self.sigma_Kk1]
		self.Kk1 = random_matrix(matrix_shape, params_Kk1, seed=self.seed_Kk1)
	
		mu_Ss0 = self.mu_Ss0
		mu_eps = self.mu_eps
		for key in kwargs:
			exec ('%s = kwargs[key]' % key)
		
		assert 0 <= self.activity_p <= 1., "Mixture ratio must be between 0 and 1"
		
		activity_mus = sp.zeros(self.Mm)
		activity_sigmas = sp.zeros(self.Mm)
		
		num_comp1 = int(self.activity_p*self.Mm)
		num_comp2 = self.Mm - num_comp1
		
		activity_mus[:num_comp1] = self.receptor_tuning_mixture_mu_1
		activity_mus[num_comp1:] = self.receptor_tuning_mixture_mu_2
		activity_sigmas[:num_comp1] = self.receptor_tuning_mixture_sigma_1
		activity_sigmas[num_comp1:] = self.receptor_tuning_mixture_sigma_2
		
		self.Kk2 = Kk2_eval_normal_activity(matrix_shape, activity_mus, 
											activity_sigmas, mu_Ss0, mu_eps, 
											self.seed_Kk2)
		
		if clip == True:
			array_dict = clip_array(dict(Kk1 = self.Kk1, Kk2 = self.Kk2))
			self.Kk1 = array_dict['Kk1']
			self.Kk2 = array_dict['Kk2']
	
	def set_measured_activity(self):
		"""
		Set the full measured activity, from nonlinear response.
		"""
		
		# Learned background firing only utilizes average background signal
		self.Yy0 = receptor_activity(self.Ss0, self.Kk1, self.Kk2, self.eps, 
									 self.binding_competitive, 
									 self.num_binding_sites)

		self.Yy = receptor_activity(self.Ss, self.Kk1, self.Kk2, self.eps, 
									self.binding_competitive,
									self.num_binding_sites)
 
		self.Yy += sp.random.normal(0, self.meas_noise, self.Mm)
		
		# Apply temporal kernel
		if self.temporal_run == False:
			pass
		else:
			kernel_params = [self.kernel_T, self.kernel_dt, self.kernel_tau_1, 
								self.kernel_tau_2, self.kernel_shape_1, 
								self.kernel_shape_2,self.kernel_alpha, 
								self.kernel_scale]
			self.Yy0, self.memory_Yy0 = temporal_kernel(self.Yy0, 
										self.memory_Yy0, self.signal_trace_Tt, 
										kernel_params)
			self.Yy, self.memory_Yy = temporal_kernel(self.Yy, 
										self.memory_Yy, self.signal_trace_Tt, 
										kernel_params)
			
		# Nonlinearities
		self.Yy0 *= self.NL_scale*(self.Yy > self.NL_threshold)
		self.Yy *= self.NL_scale*(self.Yy > self.NL_threshold)
		
		self.Yy0 = sp.minimum(self.Yy0, self.firing_max)
		self.Yy = sp.minimum(self.Yy, self.firing_max)
		
		
		# Measured response above background
		self.dYy = self.Yy - self.Yy0
	
		# Add effects of divisive normalization if called.
		if self.divisive_normalization == True:
			self.Yy0 = inhibitory_normalization(self.Yy0, self.inh_C, 
						self.inh_D, self.inh_eta, self.inh_R)
			self.Yy = inhibitory_normalization(self.Yy, self.inh_C, 
						self.inh_D, self.inh_eta, self.inh_R)
			self.dYy = self.Yy - self.Yy0
			
		
		
	######################################################
	##### 		Compressed sensing functions		 #####
	######################################################

		
		
	def set_linearized_response(self):
		"""
		Set the linearized response, which only uses the learned background. 
		This is the matrix used for CS decoding.
		"""
		
		# The linear response is scaled same as the activity.
		# Ignore the threshold here: assume that system thinks everything
		# is firing.
		self.Rr = linear_gain(self.Ss0, self.Kk1, self.Kk2, self.eps, 
							  self.binding_competitive, self. num_binding_sites)
		
		# Apply temporal kernel
		if self.temporal_run == False:
			pass
			#self.Rr *= self.kernel_scale*(1 - 2*self.kernel_alpha)
		else:
			kernel_params = [self.kernel_T, self.kernel_dt, self.kernel_tau_1, 
								self.kernel_tau_2, self.kernel_shape_1, 
								self.kernel_shape_2, self.kernel_alpha, 
								self.kernel_scale]
			self.Rr, self.memory_Rr = temporal_kernel(self.Rr, 
										self.memory_Rr, self.signal_trace_Tt, 
										kernel_params)
		
		# Apply nonlinear scaling (ignore threshold)
		self.Rr *= self.NL_scale
		
		if self.divisive_normalization == True:
			self.Rr = inhibitory_normalization_linear_gain(self.Yy0, self.Rr, 
						self.inh_C, self.inh_D, self.inh_eta, self.inh_R)
		
	def decode(self):
		"""
		Decode the response via CS.
		"""
		
		self.dSs_est = decode_CS(self.Rr, self.dYy)	
		
	def decode_nonlinear(self):
		"""
		Decode the response with nonlinear constraint on the cost function.
		"""
		
		self.dSs_est = decode_nonlinear_CS(self)
		self.dSs_est = self.dSs_est - self.Ss0

		
	
	######################################################
	#####         Temporal Coding functions          #####
	######################################################
	

	
	def set_signal_trace(self):
		"""
		Load signal trace data from file; gathered from fly walking assay.
		self.signal_trace_file is the filename of the signal trace file; 
		saved in the DATA_DIR/signal_traces folder. It is a 2-column
		tab-delimited file of which the first column is time and the second
		is the signal amplitude.
		"""
		
		assert self.signal_trace_file is not None, "Need to set "\
			"'signal_trace_file' var before calling set_signal_trace; "\
			"var should be set without extension, which must be .dat"
		
		signal_data = load_signal_trace_from_file(self.signal_trace_file)
		print('Signal time trace from file %s.dat loaded\n' \
				% self.signal_trace_file)
		self.signal_trace_Tt = signal_data[:, 0]
		self.signal_trace = (signal_data[:, 1] + self.signal_trace_offset)*\
								self.signal_trace_multiplier
		
		if self.signal_trace_file_2 is not None:
			signal_data_2 = load_signal_trace_from_file(self.signal_trace_file_2)
			print('Signal time trace 2 from file %s.dat loaded\n' \
				% self.signal_trace_file_2)
			assert len(self.signal_trace_Tt) == len(signal_data_2[:, 0]), \
				"signal_trace_file_2 must be same length as signal_trace_file"
			assert  sp.allclose(self.signal_trace_Tt, signal_data_2[:, 0], 
				1e-6), "signal_trace_file_2 must have same time array as "\
				"signal_trace_file"
			self.signal_trace_2 = (signal_data_2[:, 1] + \
									self.signal_trace_offset_2)*\
									self.signal_trace_multiplier_2
		
		# Set the estimation as a temporal run
		self.temporal_run = True
		
	def set_temporal_adapted_epsilon(self):
		"""
		Set adapted epsilon based on current value and adaptation rate.
		The adapted value is set by linear decay rate equation, 
		d(eps)/dt = beta*(a_0 - a). a_0 is set by passing the manually
		chosen variables temporal_adaptation_mu_eps and 
		temporal_adaptation_mu_Ss0 to the activity. 
		"""
		
		# Perfectly adapted activity level is set manually
		activity_stats = [self.adapted_activity_mu, self.adapted_activity_sigma]
		perfect_adapt_Yy =  random_matrix([self.Mm], params=activity_stats, 
									seed=self.seed_adapted_activity)
		
		# Make adaptation rate into a vector if it has not yet been set.
		try:
			self.temporal_adaptation_rate_vector
		except AttributeError:
			assert self.temporal_adaptation_rate_sigma == 0, "Before "\
				"setting new epsilon with set_temporal_adapted_epsilon, "\
				"you must call set_ordered_temporal_adaptation_rate, since "\
				"temporal_adaptation_rate_sigma is nonzero"
			self.temporal_adaptation_rate_vector = sp.ones(self.Mm)*\
				self.temporal_adaptation_rate
		
		# Receptor activity used for adaptation is not firing rate; just
		# get the Or/Orco activity, w/o LN functions at backend.
		current_Yy = receptor_activity(self.Ss, self.Kk1, self.Kk2, self.eps, 
									   self.binding_competitive, 
									   self.num_binding_sites)
 
		if self.temporal_adaptation_type == 'imperfect':
			d_eps_dt = (self.temporal_adaptation_rate_vector*\
						(current_Yy.T - perfect_adapt_Yy)).T
			delta_t = self.signal_trace_Tt[1] - self.signal_trace_Tt[0]
			self.eps += delta_t*d_eps_dt 
		elif self.temporal_adaptation_type == 'perfect':
			self.eps = free_energy(self.Ss, self.Kk1, self.Kk2, 
									perfect_adapt_Yy)
		
		# Enforce epsilon limits
		self.eps = sp.maximum(self.eps.T, self.min_eps).T
		self.eps = sp.minimum(self.eps.T, self.max_eps).T
			
	def set_ordered_temporal_adaptation_rate(self):
		"""
		Set a spread of adaptation rates. The spread is incorporated 
		when temporal_adaptation_rate_sigma is nonzero, 
		and this spread gives a factor change, i.e. beta -->
		beta*10^{-sigma, sigma}. 
		"""
		
		try:
			self.dYy
			self.Yy
			self.Yy0
		except AttributeError:
			print('Must run set_measured_activity(...) before calling '\
				'set_ordered_temporal_adaptation_rate(...)')
		
		sp.random.seed(self.temporal_adaptation_rate_seed)
		exp_spread = sp.random.normal(0, self.temporal_adaptation_rate_sigma, 
									  self.Mm)
		self.temporal_adaptation_rate_vector = self.temporal_adaptation_rate*\
												10.**exp_spread
		
		
		
	######################################################
	########## 		Encoding functions			##########
	######################################################

		
	
	def encode_normal_activity(self, **kwargs):
		# Run all functions to encode when activity is normally distributed.
		self.set_sparse_signals()
		self.set_normal_free_energy()
		self.set_Kk2_normal_activity(**kwargs)
		self.set_measured_activity()
		self.set_linearized_response()
	
	def encode_uniform_activity(self, **kwargs):
		# Run all functions to encode when activity is uniformly distributed.
		self.set_sparse_signals()
		self.set_normal_free_energy()
		self.set_Kk2_uniform_activity(**kwargs)
		self.set_measured_activity()
		self.set_linearized_response()
	
	def encode_normal_activity_mixture(self, **kwargs):
		# Run all functions to encode when activity arises from a mixture.
		self.set_sparse_signals()
		self.set_normal_free_energy()
		self.set_Kk2_normal_activity_mixture(**kwargs)
		self.set_measured_activity()
		self.set_linearized_response()
	
	def encode_normal_Kk(self):
		# Run all functions to encode when K matrices are Gaussian.
		self.set_sparse_signals()
		self.set_normal_free_energy()
		self.set_normal_Kk()
		self.set_measured_activity()
		self.set_linearized_response()
	
	def encode_uniform_Kk(self):
		# Run all functions to encode when K matrices are uniform.
		self.set_sparse_signals()
		self.set_normal_free_energy()
		self.set_uniform_Kk()
		self.set_measured_activity()
		self.set_linearized_response()
	
	def encode_mixture_Kk(self):
		# Run all functions to encode when full activity of each receptor 
		# is from a Gaussian mixture.
		self.set_sparse_signals()
		self.set_normal_free_energy()
		self.set_mixture_Kk()
		self.set_measured_activity()
		self.set_linearized_response()
		
	def encode_adapted_normal_activity(self):
		# Run all functions to encode when full activity of each receptor 
		# is normal.
		self.set_sparse_signals()
		self.set_normal_Kk()
		self.set_adapted_free_energy()
		self.set_measured_activity()
		self.set_linearized_response()

	def encode_manual_signal_normal_Kk(self):
		# Run all functions to encode when K matrices are normal and the 
		# signal components are manually set (e.g. for tuning curves)
		self.set_manual_signals()
		self.set_normal_Kk()
		self.set_normal_free_energy()
		self.set_measured_activity()
		self.set_linearized_response()
	
	def encode_manual_signal_uniform_Kk(self):
		# Run all functions to encode when K matrices are uniform and the 
		# signal components are manually set (e.g. for tuning curves)
		self.set_manual_signals()
		self.set_uniform_Kk()
		self.set_normal_free_energy()
		self.set_measured_activity()
		self.set_linearized_response()
	
	def encode_power_Kk(self):
		# Run all functions to encode when binding constants are 
		# taken from  a power law distribution and the energy is 
		# adapted to keep activity levels constant
		self.set_sparse_signals()
		self.set_power_Kk()
		self.set_normal_free_energy()
		self.set_measured_activity()
		self.set_linearized_response()
	
	def encode_power_Kk_adapted(self):
		# Run all functions to encode when binding constants are power
		# law distributed; can add high responders as well.
		self.set_sparse_signals()
		self.set_power_Kk()
		self.set_adapted_free_energy()
		self.set_measured_activity()
		self.set_linearized_response()
