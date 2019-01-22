"""
Aggregate CS objects from separate .pklz files to a single file.

Created by Nirag Kadakia at 22:50 08-19-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import scipy as sp
import sys
sys.path.append('../src')
from load_specs import read_specs_file
from load_data import load_objects
from save_data import save_aggregated_object_list
from utils import get_flags


def aggregate_objects(data_flags, skip_missing=False):
	"""
	Aggregate CS objects from separate .pklz files to a single .pklz file.
	
	Args:
		data_flags: Identifiers for saving and loading.
	"""

	if skip_missing == True:
		print ("Skipping missing files...will populate with `None`")

	if isinstance(data_flags, str):
		data_flags = [data_flags]

	for data_flag in data_flags:
		list_dict = read_specs_file(data_flag)
		iter_vars = list_dict['iter_vars']
		iter_vars_dims = []
		for iter_var in iter_vars:
			iter_vars_dims.append(len(iter_vars[iter_var]))		
		it = sp.nditer(sp.zeros(iter_vars_dims), flags = ['multi_index'])

		obj_list  = []
		while not it.finished:
			sys.stdout.flush()
			print (it.multi_index)
			if skip_missing == False:
				CS_obj = load_objects(list(it.multi_index), data_flag)
			else:
				try:
					CS_obj = load_objects(list(it.multi_index), data_flag)
				except (IOError, OSError):
					print ('Skipping item %s...' % list(it.multi_index))	
					CS_obj = None
			
			obj_list.append(CS_obj)
			it.iternext()

		save_aggregated_object_list(obj_list, data_flag)
		

if __name__ == '__main__':
	data_flags = get_flags()
	aggregate_objects(data_flags)
