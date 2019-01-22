"""
Functions to calculate coding capacity for different sparse signals with and
without adaptation. 

Created by Nirag Kadakia at 23:30 04-25-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
from four_state_receptor_CS import four_state_receptor_CS
from kinetics import receptor_activity, free_energy
from utils import scramble, normal_pdf
from lin_alg_structs import random_matrix


INT_PARAMS = ['Nn', 'Kk', 'Mm', 'seed_Ss0', 'seed_dSs', 'seed_Kk1', 
				'seed_Kk2', 'seed_receptor_activity', 'Kk_split', 
				'Kk_1', 'Kk_2']

class response_entropy(four_state_receptor_CS):
	"""
	Class object to calculate mutual information of an olfactory system.
	"""
	
	def __init__(self, **kwargs):
		four_state_receptor_CS.__init__(self)
		
		# Number of distinct signals to calculate entropy integral
		self.num_signals = 100
		
		# Need to have a dual signal; Kk_1 is background; Kk_2 is foreground
		# Kk_1 signal is used to determine epsilon for adaptation, Kk_1+Kk_2
		# are used for actual signal.
		self.Kk_1 = 1
		self.Kk_2 = 5
		
		# Response mesh for representing probability density at discrete points.
		# The response delta should be in Hz at firing rate level, so about 1/10
		# of meas_noise*NL_scale ~ 1e-3*3e2*1e-1 ~ 0.03
		self.response_pdf = None 
		self.response_pdf_delta = 0.05
		self.entropy = None
		
		# Overwrite variables with passed arguments	
		for key in kwargs:
			if key in INT_PARAMS:
				exec ('self.%s = int(kwargs[key])' % key)
			else:
				exec ('self.%s = kwargs[key]' % key)
		
		
	def set_signal_array(self):
		"""
		Set the array of random signals for entropy estimation.
		"""
		
		self.set_sparse_signals()
		self.Ss = sp.tile(self.Ss, reps=(self.num_signals, 1))
		self.Ss = scramble(self.Ss).T
		
	def set_mean_response_array(self):
		"""
		Set the mean receptor responses for the signal array.
		"""
		
		self.Yy = receptor_activity(self.Ss, self.Kk1, 
									self.Kk2, self.eps)
		self.Yy *= self.NL_scale*(self.Yy > self.NL_threshold)
		self.Yy = sp.minimum(self.Yy, self.firing_max)
		
	def set_response_pdf(self):
		"""
		Get the pdf p(r) using the mean responses and signal noise.
		"""
		
		self.response_mesh = sp.arange(0, self.NL_scale, 
										self.response_pdf_delta)
		self.response_pdf = sp.zeros((len(self.response_mesh), self.Mm))
		tiled_mesh = sp.tile(self.response_mesh, reps=(self.num_signals, 1))
		for iM in range(self.Mm):
			pdf_all_signals = normal_pdf(tiled_mesh.T, means=self.Yy[iM, :],
								sigmas=self.NL_scale*self.meas_noise).T
			self.response_pdf[:, iM] = sp.average(pdf_all_signals, axis=0)
			
	def calc_MI(self):
		"""
		Calculate the mutual information between signal and response.
		"""
		
		cond_H = -sp.nansum(self.response_pdf*sp.log(self.response_pdf), 
					axis=0)*self.response_pdf_delta
		noise_H = (1 + sp.log(2*sp.pi*(self.NL_scale*self.meas_noise)**2))/2
		self.entropy = (cond_H - noise_H)/sp.log(2)
		
	def encode_entropy_calc(self):
		"""
		Set the signal and epsilon values for a non-adaptive system.
		"""
		
		self.set_signal_array()
		self.set_normal_free_energy()
			
	def encode_entropy_calc_adapted(self):
		"""
		Set the signal and epsilon values for an adaptive system. Here
		mu_dSs_2 is the foreground; mu_dSs is the background.
		"""
		
		# Just do background to get adapted epsilon to background
		tmp_mu_dSs_2 = self.mu_dSs_2
		tmp_sigma_dSs_2 = self.sigma_dSs_2
		self.mu_dSs_2 = 0
		self.sigma_dSs_2 = 0
		self.set_signal_array()
		self.set_adapted_free_energy()
		
		# Now reset the Kk_2 components to re-create full fore+back signal
		self.mu_dSs_2 = tmp_mu_dSs_2
		self.sigma_dSs_2 = tmp_sigma_dSs_2
		self.set_signal_array()
		
	def encode_entropy_calc_rand_bkgrnd(self):
		"""
		Set the signal and epsilon values for non-adaptive system, given a 
		random background. Background odor is taken as mu_dSs_2; foreground
		as mu_dSs
		"""
		
		# First, set the foreground to zero; get random background
		# Store the fixed vals and seeds
		tmp_mu_dSs = self.mu_dSs
		tmp_sigma_dSs = self.sigma_dSs
		tmp_seed_Ss0 = self.seed_Ss0
		self.mu_dSs = 1e-5
		self.sigma_dSs = 0
		sp.random.seed()
		self.seed_Ss0 = sp.random.randint(1e7)
		self.set_signal_array()
		self.Ss_bck = sp.zeros(self.Ss.shape)
		self.Ss_bck[:] = self.Ss
		
		# Restore the foreground; set the background to zero
		self.mu_dSs = tmp_mu_dSs
		self.sigma_dSs = tmp_sigma_dSs
		self.seed_Ss0 = tmp_seed_Ss0
		self.mu_dSs_2 = 1e-5
		self.sigma_dSs_2 = 0
		self.set_signal_array()
		
		# Add foreground to random background to get full signal
		self.Ss += self.Ss_bck
		self.set_normal_free_energy()
		
	def encode_entropy_calc_adapted_rand_bkgrnd(self):
		"""
		Set the signal and epsilon values for an adaptive system, given a 
		random background. Background odor is taken as mu_dSs_2; foreground
		as mu_dSs
		"""
		
		# First, set the foreground to zero; get random background
		# Store the fixed vals and seeds
		tmp_mu_dSs = self.mu_dSs
		tmp_sigma_dSs = self.sigma_dSs
		tmp_seed_Ss0 = self.seed_Ss0
		self.mu_dSs = 1e-5
		self.sigma_dSs = 0
		sp.random.seed()
		self.seed_Ss0 = sp.random.randint(1e7)
		self.set_signal_array()
		self.Ss_bck = sp.zeros(self.Ss.shape)
		self.Ss_bck[:] = self.Ss
		self.set_adapted_free_energy()
		
		# Restore the foreground; set the background to zero
		self.mu_dSs = tmp_mu_dSs
		self.sigma_dSs = tmp_sigma_dSs
		self.seed_Ss0 = tmp_seed_Ss0
		self.mu_dSs_2 = 1e-5
		self.sigma_dSs_2 = 0
		self.set_signal_array()
		
		# Add foreground to random background to get full signal
		self.Ss += self.Ss_bck
		