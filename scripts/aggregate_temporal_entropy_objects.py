"""
Aggregate CS objects from for temporal entropy runs; this saves
the salient data structures rather than each CS object, 
due to space limitations and redundancy.

Created by Nirag Kadakia at 22:50 08-03-2018
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
from save_data import save_aggregated_temporal_objects
from utils import get_flags


def aggregate_temporal_entropy_objects(data_flags):
	"""
	Aggregate CS objects from separate .pklz files of temporal runs to a single
	.pklz object.
	
	Args:
		data_flags: Identifiers for saving and loading.
	"""

	temporal_structs_to_save = ['entropy']
	
	if isinstance(data_flags, str):
		data_flags = [data_flags]
	
	for data_flag in data_flags:
		list_dict = read_specs_file(data_flag)
		iter_vars = list_dict['iter_vars']
		iter_vars_dims = []
		for iter_var in iter_vars:
			iter_vars_dims.append(len(iter_vars[iter_var]))		
		it = sp.nditer(sp.zeros(iter_vars_dims), flags = ['multi_index'])
		
		CS_init_array = load_objects(list(it.multi_index), data_flag)

		# Dictionary to save all object at time 0; this will contain all 
		# non-temporal info for each iterated variable.
		data = dict()
		data['init_objs'] = []
		nT = len(CS_init_array[0].signal_trace_Tt)

		# Assign data structures of appropriate shape for the temporal variable
		structs = dict()
		for struct_name in temporal_structs_to_save:
			try:
				tmp_str = 'structs[struct_name] = CS_init_array[0].%s' \
							% struct_name
				exec(tmp_str)
			except:
				print('%s not an attribute of the CS object' % struct_name)
				continue

			# shape is (num timesteps, iterated var ranges, variable shape); 
			# if a float or integer, shape is just time and iter vars.
			struct_shape = (nT, ) +  tuple(iter_vars_dims)
			if hasattr(structs[struct_name], 'shape'):
				struct_shape += (structs[struct_name].shape)
			data['%s' % struct_name] = sp.zeros(struct_shape)

		# Iterate over all objects to be aggregated
		structs = dict()
		while not it.finished:
			
			print('Loading index:', it.multi_index)
			temporal_CS_array = load_objects(list(it.multi_index), data_flag)
			
			# Save full object at time 0, contains non-temporal data.
			data['init_objs'].append(temporal_CS_array[0])
		
			# Grab all the temporal structures, timepoint-by-timepoint
			for iT in range(nT):

				full_idx = (iT, ) + it.multi_index

				for struct_name in temporal_structs_to_save:
					tmp_str = 'structs[struct_name] = temporal_CS_array[iT].%s' \
								% struct_name
					exec(tmp_str)
					data[struct_name][full_idx] = structs[struct_name]
			
			it.iternext()

		save_aggregated_temporal_objects(data, data_flag)


if __name__ == '__main__':
	data_flags = get_flags()
	aggregate_temporal_entropy_objects(data_flags)

