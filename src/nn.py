"""
Functions to perform supervised learning of weights in full connected
olfactory circuit, from ORN to AL to MB.

Created by Nirag Kadakia at 23:30 07-11-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from four_state_receptor_CS import four_state_receptor_CS
from kinetics import receptor_activity, free_energy
from utils import tf_set_train_test_idxs
from local_methods import def_data_dir

DATA_DIR = def_data_dir()
INT_PARAMS = ['Nn', 'Kk', 'Mm', 'seed_Ss0', 'seed_dSs', 'seed_Kk1', 
				'seed_Kk2', 'seed_receptor_activity', 'Kk_split', 
				'Kk_1', 'Kk_2', 'Jj_mask_seed', 'num_signals', 
				'tf_num_classes', 'tf_num_trains', 'tf_max_steps', 
				'Zz', 'Zz_sparse', 'num_tf_ckpts']
EVAL_PARAMS = ['mu_dSs_array']

class nn(four_state_receptor_CS):
	"""
	Class object to calculate mutual information of an olfactory system.
	"""
	
	def __init__(self, **kwargs):
		four_state_receptor_CS.__init__(self)
		
		# Needed to save tf objects as model is running
		self.data_flag = None
		self.save_tf_objs = True
		self.num_tf_ckpts = 50
		
		# Various signals built from Kk_1 and Kk_2 and mu_dSs chosen from 
		# mu_dSs_array. These are aggregated and signals of same identity
		# but possibly different concentration are given same label for 
		# classification. mu_dSs_2 is set same as mu_dSs, chosen from array.
		self.Kk_1 = 1
		self.Kk_2 = 1
		self.mu_dSs_array = sp.logspace(0, 4, 10)
		self.num_signals = 100
		self.sigma_dSs = 0
		self.sigma_dSs_2 = 0
		
		# tensorflow parameters
		self.tf_max_steps = 1000
		self.tf_num_trains = 500
		self.tf_num_classes = 20
		self.tf_AL_MB_trainable = False
		self.tf_MB_read_trainable = True
		self.accuracies = None
		
		# How to split up train and test indices; can be `random` or 
		# `train_low_conc` so far.
		self.tf_idxs_shuffle_type = 'random'
		
		# ORN to PN with feedforward nonlinearity and lateral inhibition
		# Yy_PN_nonlinearity can be `lin` for no nonlinearity, `FF` for 
		#feedforward only, or `FF_LI` for both feedforward and 
		# lateral inhibition following Olsen, Bhandawat Wilson 2010
		self.Yy_PN_nonlinearity = 'lin'
		self.Yy_PN = None
		self.Yy_PN_max = 170.0
		self.Yy_PN_ff = 10.0
		self.Yy_PN_LI = 0.05
		
		# AL-->MB layer size, synaptic connectivity, and connection topology
		self.Zz = 2500
		self.Zz_sparse = 7
		self.Jj_mask = None
		self.Jj_mask_seed = 0
		
		# Overwrite variables with passed arguments	
		immutable_keys = ['mu_dSs', 'mu_dSs_2', 'sigma_dSs_2', 'sigma_dSs']
		for key in kwargs:
			
			assert hasattr(self, '%s' % key), "'%s' is not an attribute of "\
				"nn class. Check or add to __init__" % key
			
			if key in immutable_keys:
				print ('%s cannot be set for classification: instead '
						'only set the array `mu_dSs_array`. sigma_dSs, '
						'sigma_dSs_2 will be set to zero and mu_dSs_2 '
						'will be set to array values of mu_dSs_array' % key)
				quit()
			
			if key in INT_PARAMS:
				exec ('self.%s = int(kwargs[key])' % key)
			elif key in EVAL_PARAMS:
				exec ('self.%s = eval(kwargs[key])' % key)	
			else:
				exec ('self.%s = kwargs[key]' % key)
		
	def init_nn_frontend(self):
		"""
		Initialize the signal array and free energies of receptor complexes
		"""
		
		signal_array = sp.zeros((self.Nn, self.num_signals, 
								len(self.mu_dSs_array)))
		eps_array = sp.zeros((self.Mm, self.num_signals, 
								len(self.mu_dSs_array)))
	
		# Iterate to get signal and energy in [# odor IDs, # odor concs]
		it = sp.nditer(signal_array[0,...], flags=['multi_index'])
		while not it.finished:
			
			self.seed_dSs = it.multi_index[0]
			self.mu_dSs = self.mu_dSs_array[it.multi_index[1]]
			self.mu_dSs_2 = self.mu_dSs_array[it.multi_index[1]]
			
			# Set signal and random energy (and Kk matrices if not set)
			self.set_sparse_signals()
			if (self.Kk1 is None):
				self.set_power_Kk()
			self.set_normal_free_energy()
			full_idx = [slice(None), ]*len(signal_array.shape)
			full_idx[1:] = it.multi_index
			signal_array[full_idx] = self.Ss
			eps_array[full_idx] = self.eps
			
			it.iternext()
		
		# Flattened with inner loop being concentration, outer loop being ID
		# Full shape is ((# concs)*(# IDs), Nn or Mm)
		self.Ss = sp.reshape(signal_array, (self.Nn, -1))
		self.eps = sp.reshape(eps_array, (self.Mm, -1))
	
	def init_nn_frontend_adapted(self):
		"""
		Initialize the signal array and free energies of receptor complexes, 
		for a system that adapts its response to a background signal.
		"""
		
		signal_array = sp.zeros((self.Nn, self.num_signals, 
								len(self.mu_dSs_array)))
		eps_array = sp.zeros((self.Mm, self.num_signals, 
								len(self.mu_dSs_array)))
	
		# Iterate to get signal and energy in [# odor IDs, # odor concs]
		it = sp.nditer(signal_array[0,...], flags=['multi_index'])
		while not it.finished:
			
			self.seed_dSs = it.multi_index[0]
			self.mu_dSs = self.mu_dSs_array[it.multi_index[1]]
			
			# Just do background to get adapted epsilon to background
			self.mu_dSs_2 = 0
			self.set_sparse_signals()
			if (self.Kk1 is None):
				self.set_power_Kk()
			self.set_adapted_free_energy()
			
			# Now reset the Kk_2 components to re-create full fore+back signal
			self.mu_dSs_2 = self.mu_dSs_array[it.multi_index[1]]
			self.set_sparse_signals()
			
			# Now set value in signal_array and eps_array
			full_idx = [slice(None), ]*len(signal_array.shape)
			full_idx[1:] = it.multi_index
			signal_array[full_idx] = self.Ss
			eps_array[full_idx] = self.eps
			
			it.iternext()
	
		# Signal array shape = (odor-D, (# odor IDs)*(# odor intensities))
		# Flattened -- inner loop is concentration, outer loop is ID
		self.Ss = sp.reshape(signal_array, (self.Nn, -1))
		self.eps = sp.reshape(eps_array, (self.Mm, -1))
	
	def set_ORN_response_array(self):
		"""
		Set the ORN responses for the signal array.
		"""
		
		self.Yy = receptor_activity(self.Ss, self.Kk1, 
									self.Kk2, self.eps)
		self.Yy *= self.NL_scale*(self.Yy > self.NL_threshold)
		self.Yy = sp.minimum(self.Yy, self.firing_max)
	
	def set_PN_response_array(self):
		"""
		Set PN response array, including feedforward nonlinearity and 
		lateral inhibition
		"""
		
		if self.Yy_PN_nonlinearity == 'lin':
			self.Yy_PN = self.Yy
		elif self.Yy_PN_nonlinearity == 'FF':
			self.Yy_PN = self.Yy_PN_max*self.Yy**(1.5)/(self.Yy**1.5 
							+ self.Yy_PN_ff**1.5)
		elif self.Yy_PN_nonlinearity == 'FF_LI':
			s_ORN = sp.sum(self.Yy, axis=0)
			self.Yy_PN = self.Yy_PN_max*self.Yy**(1.5)/(self.Yy**1.5 
							+ self.Yy_PN_ff**1.5 + (self.Yy_PN_LI*s_ORN)**1.5)
		else:
			print ("PN_nonlinearity %s not accepted; must be `FF_LI`, `FF`, "
					"or `lin`" % self.Yy_PN_nonlinearity)
			quit()
			
	def set_AL_MB_connectome(self):
		"""
		Set the connection topology -- this is not mutable during training.
		"""
		
		self.Jj_mask = sp.zeros((self.Mm, self.Zz))
		sp.random.seed(self.Jj_mask_seed)
		for iZ in range(self.Zz):
			idxs = sp.random.choice(range(self.Mm), self.Zz_sparse, replace=0)
			self.Jj_mask[idxs, iZ] = 1
		
	def init_tf(self):
		"""
		Initialize all the tensorflow variables for training.
		"""
		
		assert self.Jj_mask is not None, "Set Al-->MB connectome first with "\
			"set_AL_MB_connectome(..)"
		
		tf.reset_default_graph()
		self.tf_input = tf.placeholder(tf.float32, shape=[None, self.Mm])
		self.tf_labels = tf.placeholder(tf.float32, 
							shape=[None, self.tf_num_classes])
		
		# AL --> MB (Mm ORNs to Zz KCs) up to user if can be trained or not
		self.tf_J1 = tf.Variable(tf.random_normal(shape=[self.Mm, self.Zz], 
				mean=0., stddev=1./sp.sqrt(self.Zz_sparse)), 
				trainable=self.tf_AL_MB_trainable)
		
		# MB --> readout (Zz KCs to num_classes) can be adjusted in training
		self.tf_J2 = tf.Variable(tf.random_normal(shape=[self.Zz, 
						self.tf_num_classes], mean=0, stddev=1./sp.sqrt(self.Zz)),
						trainable=self.tf_MB_read_trainable)

		# MB--> read connections can be adjusted in training
		self.tf_AL_MB_bias = tf.Variable(tf.zeros([1, self.Zz]), 
								trainable=self.tf_AL_MB_trainable)
		self.tf_MB_read_bias = tf.Variable(tf.zeros(self.tf_num_classes), 
								trainable=self.tf_MB_read_trainable)
		self.tf_MB = tf.nn.relu(tf.matmul(self.tf_input, 
								self.Jj_mask*self.tf_J1)) + self.tf_AL_MB_bias
		self.tf_calc_output =  tf.matmul(self.tf_MB, self.tf_J2) \
								+ self.tf_MB_read_bias
		
		# Cost function to reduce cross entropy (logistic regression) and error
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=
								self.tf_calc_output, labels=self.tf_labels))
		self.tf_train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
		correct_pred = tf.equal(tf.argmax(self.tf_calc_output, 1), 
								tf.argmax(self.tf_labels, 1))
		self.tf_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	def set_tf_class_labels(self):
		"""
		Set the class labels for each signal, both for training and testing.
		"""
	
		num_concs = len(self.mu_dSs_array)
		tmp_labels = sp.zeros((self.num_signals, self.tf_num_classes))
		for iS in range(self.num_signals):
		
			# Consecutive signals get consecutive class names. Set 
			# self.num_classes == self.num_signals to get unique class 
			# for each signal.
			tmp_labels[iS, iS % self.tf_num_classes] = 1
		
		# Labels are same for different concentrations of a given odor ID
		self.labels = sp.repeat(tmp_labels, num_concs, axis=0)
		
		# Split indices into training and testing sets
		train_idxs, test_idxs = tf_set_train_test_idxs(num_concs, 
								self.num_signals, self.tf_num_trains, 
								self.tf_idxs_shuffle_type)
		
		self.train_data_labels = self.labels[train_idxs, :]
		self.train_data_in = self.Yy_PN.T[train_idxs, :]
		self.test_data_labels = self.labels[test_idxs, :]
		self.test_data_in = self.Yy_PN.T[test_idxs, :]
		
	def train_and_test_tf(self):
		"""
		Train and test the classifier. 
		"""
		
		# Initialize a new tensorflow class
		tf_sess = tf.InteractiveSession()
		tf_sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep=self.num_tf_ckpts)
	
		# Train and calculate tf_accuracy at each training step.
		self.accuracies = sp.zeros(self.tf_max_steps)
		
		print ("Training and testing network...\n")
		for iStep in range(self.tf_max_steps):
			print ('\r', 100*iStep/self.tf_max_steps, '%', end='')
			
			tf_sess.run(self.tf_train_step, 
						feed_dict={self.tf_input: self.train_data_in, 
						self.tf_labels: self.train_data_labels})
			
			acc = tf_sess.run(self.tf_accuracy, 
								feed_dict={self.tf_input: self.test_data_in, 
								self.tf_labels: self.test_data_labels})
			self.accuracies[iStep] = acc
			
			if self.save_tf_objs is True:
				if iStep % (self.tf_max_steps/self.num_tf_ckpts) == 0:	
					out_dir = '%s/objects/%s/ckpt' % (DATA_DIR, self.data_flag)
					saver.save(tf_sess, out_dir, global_step=iStep)
		
		# Generate predicted labels for test data after training
		self.test_data_calc = tf_sess.run(self.tf_calc_output, 
								feed_dict={self.tf_input: self.test_data_in})
	
	def del_tf_vars(self):
		"""
		Delete all tensor flow variables to allow object pickling. This is 
		super hacky.
		"""
		
		vars_to_del = []
		for var in vars(self):
			if "tf_" in var:
				vars_to_del.append('self.%s' % var)
		for var in vars_to_del:
			exec("del %s" % var)
		