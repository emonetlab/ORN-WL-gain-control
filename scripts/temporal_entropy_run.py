"""
Run an MI calculation in time. Treats each i/o as the adapted response
to that given signal. Thus each signal demands a different epsilon array
at a given point in time.

Created by Nirag Kadakia at 22:26 08-03-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import sys
import copy
sys.path.append('../src')
from entropy import response_entropy
from utils import get_flag
from save_data import dump_objects
from load_specs import read_specs_file, compile_all_run_vars


def temporal_entropy_run(data_flag, iter_var_idxs, 
					mu_dSs_offset=0, mu_dSs_multiplier=1./3., 
					sigma_dSs_offset=0, sigma_dSs_multiplier=1./9., 
					signal_window=None, save_data=True):
	
	assert mu_dSs_offset >= 0, "mu_dSs_offset kwarg must be >= 0"
	assert sigma_dSs_offset >= 0, "sigma_dSs_offset kwarg must be >= 0"
	
	# Aggregate all run specifications from the specs file; instantiate model
	list_dict = read_specs_file(data_flag)
	if 'run_type' in list_dict['run_specs'].keys():
		print ('!!\n\nrun_spec %s passed in specs file. run_specs are not '
				'accepted for temporal entropy calculations at this time. '
				'Ignoring...\n\n!!\n' % list_dict['run_specs']['run_type'])
	vars_to_pass = compile_all_run_vars(list_dict, iter_var_idxs)
	obj = response_entropy(**vars_to_pass)
	obj.encode_power_Kk()	
	
	# Set the temporal signal array from file; truncate to signal window
	obj.set_signal_trace()
	
	assert sp.sum(obj.signal_trace <= 0) == 0, \
		"Signal contains negative values; increase signal_trace_offset"
	if signal_window is not None:
		obj.signal_trace_Tt = obj.signal_trace_Tt[signal_window[0]: \
													signal_window[1]]
		obj.signal_trace = obj.signal_trace[signal_window[0]: signal_window[1]]
	
	# Load dual odor dSs from file (this is considered non-adapted fluctuation
	# and should have a shorter timescale than the first odor). Can also use
	# Kk_1 and Kk_2 for separate complexities of odor 1 and 2, respectively.
	if (obj.Kk_1 is not None) and (obj.Kk_2 is not None):
		obj.Kk = obj.Kk_1 + obj.Kk_2
		obj.Kk_split = obj.Kk_2
	
	if (obj.Kk_split is not None) and (obj.Kk_split != 0):
		try: 
			obj.signal_trace_2
		except AttributeError:
			print('Need to assign signal_trace_2 if setting Kk_split or ' \
					'Kk_1 and Kk_2 nonzero') 
			quit()
		assert sp.sum(obj.signal_trace_2 <= 0) == 0, \
				"Signal_2 contains neg values; increase signal_trace_offset_2"
		if signal_window is not None:
			obj.signal_trace_2 = obj.signal_trace_2[signal_window[0]: \
													signal_window[1]]
	
	obj_list = []
	
	for iT, dt in enumerate(obj.signal_trace_Tt):
		print('%s/%s' % (iT + 1, len(obj.signal_trace)), end=' ')
		sys.stdout.flush()
		
		# Set mu_Ss0 from signal trace, if desired
		if obj.set_mu_Ss0_temporal_signal == True:
			obj.mu_Ss0 = obj.signal_trace[iT]
		
		# Set estimation dSs values from signal trace and kwargs
		signal = obj.signal_trace[iT]
		obj.mu_dSs = mu_dSs_offset + signal*mu_dSs_multiplier
		obj.sigma_dSs = sigma_dSs_offset + signal*sigma_dSs_multiplier
		
		# Set estimation dSs values for dual odor if needed
		if (obj.Kk_split is not None) and (obj.Kk_split != 0):
			signal_2 = obj.signal_trace_2[iT]
			obj.mu_dSs_2 = mu_dSs_offset + signal_2*mu_dSs_multiplier
			obj.sigma_dSs_2 = sigma_dSs_offset + signal_2*sigma_dSs_multiplier
		
		# Set the full signal array from the above signal parameters
		obj.set_signal_array()
		
		# At first step, set energy; from then on it is dynamical.
		if iT == 0:
			obj.set_normal_free_energy()
			
			# Spread adaptation rates over the system
			if obj.temporal_adaptation_rate_sigma != 0:
				obj.set_ordered_temporal_adaptation_rate()
		else:
			obj.set_temporal_adapted_epsilon()
			
		# Calculate MI
		obj.set_mean_response_array()
		obj.set_response_pdf()
		obj.calc_MI()
		
		# Deep copy to take all aspects of the object but not update it
		obj_list.append(copy.deepcopy(obj))
	
	if save_data == True:
		dump_objects(obj_list, iter_var_idxs, data_flag)
	
	return obj_list
	
	
if __name__ == '__main__':
	data_flag = get_flag()
	iter_var_idxs = list(map(int, sys.argv[2:]))
	temporal_entropy_run(data_flag, iter_var_idxs)
