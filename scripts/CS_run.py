"""
Run a CS decoding run for one given index of a set of iterated
variables. 

Created by Nirag Kadakia at 14:40 08-17-2017
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
from four_state_receptor_CS import four_state_receptor_CS
from utils import get_flag
from save_data import dump_objects
from load_specs import read_specs_file, compile_all_run_vars
from encode_CS import single_encode_CS


def CS_run(data_flag, iter_var_idxs):
	"""
	Run a CS decoding run for one given index of a set of iterated
	variables. 

	Data is read from a specifications file in the data_dir/specs/ 
	folder, with proper formatting given in read_specs_file.py. The
	specs file indicates the full range of the iterated variable; this
	script only produces output from one of those indices, so multiple
	runs can be performed in parallel.
	"""
	
	# Aggregate all run specifications from the specs file; instantiate model
	list_dict = read_specs_file(data_flag)
	vars_to_pass = compile_all_run_vars(list_dict, iter_var_idxs)
	obj = four_state_receptor_CS(**vars_to_pass)
	
	# Encode and decode
	obj = single_encode_CS(obj, list_dict['run_specs'])
	obj.decode()
	
	dump_objects(obj, iter_var_idxs, data_flag)

	
if __name__ == '__main__':
	data_flag = get_flag()
	iter_var_idxs = list(map(int, sys.argv[2:]))
	CS_run(data_flag, iter_var_idxs)
