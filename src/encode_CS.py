"""
Run CS encoding and decoding via four_state_receptor_CS; single iteration.


Created by Nirag Kadakia at 10:30 09-05-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


def single_encode_CS(obj, run_specs=dict()):
	"""
	Run CS encoding and decoding via four_state_receptor_CS; single iteration.
	Reads run_specs and performs appropriate encoding scheme depending on 
	specs input.
	
	Args:
		obj: four_state_receptor_CS.py class object 
		run_specs: dictionary; parameters of the run.
		
	Returns:
		obj: now with variables updated via encoding.
	"""		
	
	if 'run_type' in list(run_specs.keys()):
		val = run_specs['run_type']
		if val[0] == 'normal_activity_fixed_Kk2':
			override_parameters = dict()
			override_parameters['mu_Ss0'] = float(val[1])
			override_parameters['mu_eps'] = float(val[2])
			obj.encode_normal_activity(**override_parameters)
		elif val[0] == 'uniform_activity_fixed_Kk2':
			override_parameters = dict()
			override_parameters['mu_Ss0'] = float(val[1])
			override_parameters['mu_eps'] = float(val[2])
			obj.encode_uniform_activity(**override_parameters)		
		else:
			try: 
				hasattr(obj, 'encode_%s' % val[0]) and \
				callable(getattr(obj, 'encode_%s' % val[0]))
			except AttributeError:
				print(('Run specification %s not recognized' % val[0]))
				quit()
			str = 'obj.encode_%s()' % val[0]
			exec(str)
	else:
		print ('No run type specified, proceeding with normal_activity')
		obj.encode_normal_activity()
	
	return obj