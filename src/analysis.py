"""
Error quantification, analysis, etc. methods. 

Created by Nirag Kadakia at 18:00 09-07-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp

def binary_errors(CS_object, nonzero_bounds=[0.7, 1.3], zero_bound=1./25):

	Nn = CS_object.Nn
	mu_dSs = CS_object.mu_dSs
	sparse_idxs =  CS_object.idxs[0]

	errors_nonzero = 0
	errors_zero = 0
	
	for iN in range(Nn):
		if iN in sparse_idxs: 
			scaled_estimate = 1.*CS_object.dSs_est[iN]/CS_object.dSs[iN]
			if nonzero_bounds[0] < scaled_estimate < nonzero_bounds[1]:
				errors_nonzero += 1
		else:
			if abs(CS_object.dSs_est[iN]) <  abs(mu_dSs*zero_bound):
				errors_zero += 1

	errors = dict()
	errors['errors_nonzero'] = sp.around(1.*errors_nonzero/ \
											len(sparse_idxs)*100., 2)
	errors['errors_zero'] = sp.around(1.*errors_zero/ \
											(Nn - len(sparse_idxs))*100., 2)
										
	return errors

def binary_errors_temporal_run(init_CS_object, dSs, dSs_est, mu_dSs, 
								nonzero_bounds=[0.7, 1.3], zero_bound=1./10, 
								dual=False):
	
	# Last index is the actual stimulus vector; first index is timepoint
	Nn = init_CS_object.Nn
	sparse_idxs =  init_CS_object.idxs[0]
	if dual == True:
		idxs_2 = init_CS_object.idxs_2
		idxs_1 = []
		for idx in sparse_idxs:
			if idx not in idxs_2:
				idxs_1.append(idx)
	
	nT = dSs.shape[0]
	
	# Check dimension of the stimuli
	assert len(dSs.shape) == 2, "Need to pass rank-2 tensor for dSs; first "\
								"index is time, second is Nn"
	assert len(dSs_est.shape) == 2, "Need to pass rank-2 tensor for dSs_est; "\
								"first index is time, second is Nn"
	assert len(mu_dSs.shape) == 1, "Need to pass 1-rank array for mu_dSs"
	assert len(mu_dSs) == nT, "mu_dSs must be length nT=%s" % nT
	
	errors_nonzero = sp.zeros(nT)
	errors_zero = sp.zeros(nT)
	
	for iN in range(Nn):
		if iN in sparse_idxs: 
			if (dual == True) and (iN in idxs_2):
				continue
			scaled_estimate = 1.*dSs_est[:, iN]/dSs[:, iN]
			errors_nonzero += (nonzero_bounds[0] < scaled_estimate)*\
								(scaled_estimate < nonzero_bounds[1])
		else:
			zero_est = (sp.absolute(dSs_est[:, iN]) < abs(mu_dSs*zero_bound))
			errors_zero += zero_est

	errors = dict()
	if dual == True:
		errors['errors_nonzero'] = sp.around(1.*errors_nonzero/ \
								len(idxs_1)*100., 2)
	else:
		errors['errors_nonzero'] = sp.around(1.*errors_nonzero/ \
								len(sparse_idxs)*100., 2)
	errors['errors_zero'] = sp.around(1.*errors_zero/ \
							(Nn - len(sparse_idxs))*100., 2)
										
	return errors
	
def MSE_errors(CS_object):

	Nn = CS_object.Nn
	sparse_idxs =  CS_object.idxs[0]

	errors_nonzero = 0
	errors_zero = 0
	
	for iN in range(Nn):
		if iN in sparse_idxs: 
			errors_nonzero += (CS_object.dSs[iN] - CS_object.dSs_est[iN])**2.0
		else:
			errors_zero += (CS_object.dSs[iN] - CS_object.dSs_est[iN])**2.0
	
	errors = dict()
	errors['errors_nonzero'] = errors_nonzero/len(sparse_idxs)
	errors['errors_zero'] = errors_zero/(Nn - len(sparse_idxs))
										
	return errors

def binary_success(errors_nonzero, errors_zero, threshold_pct_nonzero=100.0, 
					threshold_pct_zero=100.0):

	success = (errors_zero >= threshold_pct_zero)*\
				(errors_nonzero >= threshold_pct_nonzero)
		
	return success
	
def binary_errors_dual_odor(CS_object, nonzero_bounds=[0.7, 1.3], 
							zero_bound=1./25):
	
	Nn = CS_object.Nn
	mu_dSs = CS_object.mu_dSs
	mu_dSs_2 = CS_object.mu_dSs_2
	sparse_idxs =  CS_object.idxs[0]
	idxs_2 =  CS_object.idxs_2
	
	errors_nonzero = 0
	errors_nonzero_2 = 0
	errors_zero = 0
	errors_zero_2 = 0
	
	for iN in range(Nn):
		if iN in sparse_idxs: 
			if iN in idxs_2:
				scaled_estimate = 1.*CS_object.dSs_est[iN]/CS_object.dSs[iN]
				if nonzero_bounds[0] < scaled_estimate < nonzero_bounds[1]:
					errors_nonzero_2 += 1
			else:
				scaled_estimate = 1.*CS_object.dSs_est[iN]/CS_object.dSs[iN]
				if nonzero_bounds[0] < scaled_estimate < nonzero_bounds[1]:
					errors_nonzero += 1
		else:
			if abs(CS_object.dSs_est[iN]) <  abs(mu_dSs*zero_bound):
				errors_zero += 1
			if CS_object.Kk_split != 0:
				if abs(CS_object.dSs_est[iN]) <  abs(mu_dSs_2*zero_bound):
					errors_zero_2 += 1
			
	errors = dict()
	
	# Save errors; special cases if split is 0 or full
	if set(idxs_2) == set(sparse_idxs):
		errors['errors_nonzero'] = 0
	else:
		errors['errors_nonzero'] = \
			sp.around(1.*errors_nonzero/(len(sparse_idxs) - len(idxs_2))*100, 2)
	if len(idxs_2) == 0:
		errors['errors_nonzero_2'] = 0
	else:
		errors['errors_nonzero_2'] = \
			sp.around(1.*errors_nonzero_2/len(idxs_2)*100., 2)
	errors['errors_zero'] = \
		sp.around(1.*errors_zero/(Nn - len(sparse_idxs))*100., 2)
	errors['errors_zero_2'] = \
		sp.around(1.*errors_zero_2/(Nn - len(sparse_idxs))*100., 2)
										
	return errors