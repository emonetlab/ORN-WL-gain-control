"""
Functions for loading data for analysis.

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
from local_methods import def_data_dir


DATA_DIR = def_data_dir()

def load_Hallem_firing_rate_data():
	"""
	Load firing rate data from file.
	
	Returns:
		data: dictionary of 24 ORN firing rates for each of 
				110 monomolecular odorants.
	"""
	
	filename = '%s/Hallem_data/firing_rates.dat' % (DATA_DIR)
	data = sp.loadtxt(filename, dtype='S8')
	unicode_data = data.view(sp.chararray).encode('utf-8')
		
	data_dict = dict()
	ORN_names = data[0, :]
	ORN_rates = sp.array(data[1:, :], dtype='float').T
	for idx, name in enumerate(ORN_names): 
		data_dict[name] = (ORN_rates[idx])
	
	return data_dict					

def load_objects(obj_idx, data_flag):
	"""
	Load objects saved by save_objects in the save_data module. 

	Args:
		obj_idx: List whose indices corresponding to name of saved *.npz 
					file holding object array.
		data_flag: Data identifier for loading and saving.
	"""

	filename = '%s/objects/%s/%s.pklz' % (DATA_DIR, data_flag, obj_idx)
	with gzip.open(filename, 'rb') as f:
		obj = pickle.load(f)
	
	return obj
	
def load_aggregated_object_list(iter_vars_dims, data_flag):
	"""
	Load objects saved by aggregated_objects module; in correct array
	reflecting the shape of the iterated variables.

	Args:
		iter_vars_dims: list of dimensions of iterated variables
		data_flag: Data identifier for loading and saving.
	
	Returns:
		agg_obj: Array of objects
	"""

	filename = '%s/objects/%s/aggregated_objects.pklz' % (DATA_DIR, data_flag)
	with gzip.open(filename, 'rb') as f:
		CS_object_list = pickle.load(f)
	
	CS_object_array = sp.reshape(CS_object_list, iter_vars_dims)

	return CS_object_array

def load_MSE_errors(data_flag):
	"""
	Load .npz file containing error data.

	Args:
		data_flag: Data identifier for loading and saving.
	
	Returns:
		errors: Numpy object containing error data.
	"""

	filename = '%s/analysis/%s/MSE_errors.npz' % (DATA_DIR, data_flag)
	errors = dict()
	infile = sp.load(filename)
	errors['errors_nonzero'] = infile['errors_nonzero']
	errors['errors_zero'] = infile['errors_zero']
	
	return errors
	
def load_binary_errors(data_flag):
	"""
	Load .npz file containing error data.

	Args:
		data_flag: Data identifier for loading and saving.
	
	Returns:
		errors: Numpy object containing error data.
	"""
	
	filename = '%s/analysis/%s/binary_errors.npz' % (DATA_DIR, data_flag)
	errors = dict()
	infile = sp.load(filename)
	errors['errors_nonzero'] = infile['errors_nonzero']
	errors['errors_zero'] = infile['errors_zero']
	
	return errors

def load_binary_errors_dual_odor(data_flag):
	"""
	Load .npz file containing error data.

	Args:
		data_flag: Data identifier for loading and saving.
	
	Returns:
		errors: Numpy object containing error data.
	"""
	
	filename = '%s/analysis/%s/binary_errors.npz' % (DATA_DIR, data_flag)
	errors = dict()
	infile = sp.load(filename)
	errors['errors_nonzero'] = infile['errors_nonzero']
	errors['errors_nonzero_2'] = infile['errors_nonzero_2']
	errors['errors_zero'] = infile['errors_zero']
	
	return errors
	
def load_success_ratios(data_flag):
	"""
	Load list of successes based on decoding error of CS
	objects.
	
	Args:
		data_flag: Data identifier for loading and saving.
		
	Returns:
		successes: numpy object of success ratio data
	"""
	
	filename = '%s/analysis/%s/successes.npz' % (DATA_DIR, data_flag)
	infile = sp.load(filename)
	successes = infile['successes']
	
	return successes
	
def load_signal_trace_from_file(file):
	"""
	Load a signal trace from file. File should be a two column array, 
	first column is time in seconds, second column is signal amplitude
	"""
	
	filename = '%s/signal_traces/%s.dat' % (DATA_DIR, file)
	
	signal_trace = sp.loadtxt(filename, dtype=float)
	
	return signal_trace
	
def load_aggregated_temporal_objects(data_flag):
	"""
	Load dictionary object holding all info from a batch temporal CS run
	
	Args:
		data_flag: Data identifier for loading and saving.
		
	Returns:
		data_dict: dictionary; one key is 'init_objs', which holds the CS obj
			at the first time point of one of the iterated variables; the 
			remaining keys are the variables of interest for each temporal
			point for each iterated variable.
					
	"""
	
	filename = '%s/objects/%s/aggregated_temporal_objects.pklz' \
				% (DATA_DIR, data_flag)
	with gzip.open(filename, 'rb') as f:
		data_dict = pickle.load(f)
	
	return data_dict
	
def load_aggregated_entropy_objects(data_flag):
	"""
	Load dictionary object holding all info from a batch entropy calculation
	
	Args:
		data_flag: Data identifier for loading and saving.
		
	Returns:
		data_dict: dictionary of data
					
	"""
	
	filename = '%s/objects/%s/aggregated_entropy_objects.pklz' \
				% (DATA_DIR, data_flag)
	with gzip.open(filename, 'rb') as f:
		data_dict = pickle.load(f)
	
	return data_dict