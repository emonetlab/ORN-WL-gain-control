"""
Functions for saving data for later analysation

Created by Nirag Kadakia at 23:30 08-02-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import pickle
import shelve
import gzip
import os
import time
try:
	import matplotlib.pyplot as plt
except:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
from local_methods import def_data_dir, def_analysis_dir


DATA_DIR = def_data_dir()
ANALYSIS_DIR = def_analysis_dir()


def save_MSE_errors(errors_nonzero, errors_zero, data_flag):
	"""
	Save decoding error from array of CS objects as numpy object.

	Args:
		errors: Error array to be saved
		data_flag: Data identifier for saving and loading.
	"""

	out_dir = '%s/analysis/%s' % (DATA_DIR, data_flag)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	filename = '%s/MSE_errors.npz' % out_dir
	sp.savez(filename, errors_nonzero=errors_nonzero, errors_zero=errors_zero)
	print('\nSignal errors file saved to %s' % filename)
			
def save_binary_errors(errors_nonzero, errors_zero, data_flag):
	"""
	Save decoding error from array of CS objects as numpy object, 
	above or below certain threshold for nonzero and zero components.

	Args:
		errors_nonzero_components: Error array to be saved, for 
			nonzero components
		errors_zero_components: Error array to be saved, for 
			zero components
		data_flag: Data identifier for saving and loading.
	"""

	out_dir = '%s/analysis/%s' % (DATA_DIR, data_flag)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	filename = '%s/binary_errors.npz' % out_dir
	sp.savez(filename, errors_nonzero=errors_nonzero, errors_zero=errors_zero)
	print('\nSignal errors file saved to %s' % filename)

def save_binary_errors_dual_odor(errors_nonzero, errors_nonzero_2, 
									errors_zero, data_flag):
	"""
	Save decoding error from array of CS objects as numpy object, 
	above or below certain threshold for nonzero and zero components.

	Args:
		errors_nonzero_components: Error array to be saved, for 
			nonzero components of sparse_idxs not in idxs_2
		errors_nonzero_components: Error array to be saved, for 
			nonzero components of idxs_2
		errors_zero_components: Error array to be saved, for 
			zero components
		data_flag: Data identifier for saving and loading.
	"""

	out_dir = '%s/analysis/%s' % (DATA_DIR, data_flag)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	filename = '%s/binary_errors.npz' % out_dir
	sp.savez(filename, errors_nonzero=errors_nonzero, 
				errors_nonzero_2=errors_nonzero_2, 
				errors_zero=errors_zero)
	print('\nSignal errors file saved to %s' % filename)
	
def save_figure(fig, suffix, data_flag, clear_plot=True):
	"""
	Save a generic figure.
	
	Args: 
		fig: Figure object to be saved.
		suffix: Type of figure. Ex: 'error_plot'.
		data_flag: Data identifier for saving and loading.
		clear_plot: binary; if True, clear figure window.
	"""
	
	out_dir = '%s/figures/%s' % (ANALYSIS_DIR, data_flag)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	
	filename = '%s/%s_%s.pdf' %(out_dir, suffix, data_flag)
	plt.tight_layout()
	plt.savefig(filename, bbox_inches = 'tight')
	
	if clear_plot == True:
		plt.close()

def dump_objects(CS_obj, iter_vars_idxs, data_flag, output=True):
	"""
	Save object instantiation from CS decoder as pickled object.
	
	Args:
		CS_obj: The instantiated four_state_receptor_CS object
		iter_vars_idxs: Arguments of iterated variable indices from command 
						line arguments.
		data_flag: Data identifier for loading and saving
	"""
	
	out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	
	filename = '%s/%s.pklz' % (out_dir, iter_vars_idxs)
	with  gzip.open(filename, 'wb') as f:
		pickle.dump(CS_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
	if output == True:
		print("\n -- Object array item %s saved." % str(iter_vars_idxs))

def save_aggregated_object_list(agg_obj_list, data_flag):
	"""
	Save list of aggregated objects to file.
	
	Args:
		agg_obj_list: List of four_state_receptor_CS objects.
		data_flag: Data identifier for loading and saving.
	"""
	
	filename = '%s/objects/%s/aggregated_objects.pklz' % (DATA_DIR, data_flag)

	with gzip.open(filename, 'wb') as f:
		pickle.dump(agg_obj_list, f, protocol=pickle.HIGHEST_PROTOCOL)

	print('Aggregated object file %s saved.' % filename)

def save_aggregated_temporal_objects(agg_obj_dict, data_flag):
	"""
	Save the dictionary of aggregated objects in a temporal CS run.

	Args: 
		agg_obj_dict: dictionary; one key should be 'init_objs' which is a list 
			holding holds each full CS object of the first timepoint for every 
			iterated variable. The remaining keys are numpy 
			arrays holding different temporal variables, indexed by 
			(timepoint, iterated variables, variable dimension)
	"""

	filename = '%s/objects/%s/aggregated_temporal_objects.pklz' \
				% (DATA_DIR, data_flag)

	with gzip.open(filename, 'wb') as f:
		pickle.dump(agg_obj_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

	print('Aggregated temporal object file %s saved.' % filename)

def save_aggregated_entropy_objects(agg_obj_dict, data_flag):
	"""
	Save the dictionary of aggregated objects in an entropy calculation run.

	Args: 
		agg_obj_dict: dictionary; one key should be 'init_objs' which is a list
			holding holds each full CS object of the first timepoint for every 
			iterated variable. The remaining keys are numpy 
			arrays holding different temporal variables, indexed by 
			(timepoint, iterated variables, variable dimension)
	"""

	filename = '%s/objects/%s/aggregated_entropy_objects.pklz' \
				% (DATA_DIR, data_flag)

	with gzip.open(filename, 'wb') as f:
		pickle.dump(agg_obj_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

	print('Aggregated entropy calculation object file %s saved.' % filename)

def save_success_ratios(successes, data_flag):
	"""
	Save list of successes based on decoding error of CS
	objects.
	
	Args:
		successes: numpy array of number of binary data for
					success (1) or not success (0), for full CS
					object array.
		data_flag: Data identifier for loading and saving.
	"""
	
	out_dir = '%s/analysis/%s' % (DATA_DIR, data_flag)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	filename = '%s/successes.npz' % out_dir
	sp.savez(filename, successes=successes)
	print('\nSignal binary successes file saved to %s' % filename)
	
