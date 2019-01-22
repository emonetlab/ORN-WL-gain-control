"""
Train the downstream olfactory network as a multi-class classifier.

Created by Nirag Kadakia at 13:40 07-11-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import sys
import os
sys.path.append('../src')
from nn import nn
from utils import get_flag
from load_specs import read_specs_file, compile_all_run_vars
from save_data import dump_objects

def nn_run(data_flag, iter_var_idxs):
	"""
	Run a supervised learning classification for a single specs file.

	Data is read from a specifications file in the data_dir/specs/ 
	folder, with proper formatting given in read_specs_file.py. The
	specs file indicates the full range of the iterated variable; this
	script only produces output from one of those indices, so multiple
	runs can be performed in parallel.
	"""
	
	# Aggregate all run specifications from the specs file; instantiate model
	list_dict = read_specs_file(data_flag)
	vars_to_pass = compile_all_run_vars(list_dict, iter_var_idxs)
	obj = nn(**vars_to_pass)
	
	# Need this to save tensor flow objects on iterations
	obj.data_flag = data_flag
	
	# Set the signals and free energy, depending if adaptive or not.
	if 'run_type' in list(list_dict['run_specs'].keys()):
		val = list_dict['run_specs']['run_type']
		if val[0] == 'nn':
			obj.init_nn_frontend()
		elif val[0] == 'nn_adapted':
			obj.init_nn_frontend_adapted()
		else:
			print('`%s` run type not accepted for '
					'supervised learning calculation' % val[0])
			quit()
	else:
		print ('No learning calculation run type specified, proceeding with' \
				'unadapted learning calculation')
		obj.init_nn_frontend()
	
	# Set the network variables, learning algorithm
	obj.set_AL_MB_connectome()
	obj.set_ORN_response_array()
	obj.set_PN_response_array()
	obj.init_tf()
	
	# Train and test performance
	obj.set_tf_class_labels()
	obj.train_and_test_tf()
		
	# Delete tensorflow variables to allow saving
	obj.del_tf_vars()
	dump_objects(obj, iter_var_idxs, data_flag)
	
	return obj
	
	
if __name__ == '__main__':
	data_flag = get_flag()
	iter_var_idxs = list(map(int, sys.argv[2:]))
	nn_run(data_flag, iter_var_idxs)
