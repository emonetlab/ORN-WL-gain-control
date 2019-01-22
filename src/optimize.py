"""
Module holding methods for optimizing for CS decoding.


Created by Nirag Kadakia at 10:30 09-05-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
from scipy.optimize import minimize


def decode_CS(Rr, Yy, opt_type="L1_strong", precision='None', 
				init_params=[0, 1]):
	"""
	Run CS decoding with L1 norm.
	
	Args:
		Rr: numpy array; measurement matrix.
		Yy: numpy array; Measured signal.
	
	Optional args:
		opt_type: String for type of L1 minimization "L1_strong" or "L1_weak".
		precision: Float, for L1_weak, the multiplier of the squared error.
		init_params: List; initialization point for the optimization.
	"""		

	def L1_strong(x):
		return sp.sum(abs(x))

	def L1_weak(x,*args):
		Rr, Yy, precision = args
		tmp1 = sp.sum(abs(x))
		tmp2 = precision*sp.sum((sp.dot(Rr, x) - Yy)**2.0)
		return tmp1+tmp2

	Nn = len(Rr[0,:])
	
	if opt_type == "L1_strong":
		constraints = ({'type': 'eq', 'fun': lambda x: sp.dot(Rr, x) - Yy})
		res = minimize(L1_strong, 
						sp.random.normal(init_params[0], init_params[1], Nn), 
						method='SLSQP', constraints = constraints)
	elif opt_type == "L1_weak":
		res = minimize(L1_weak, 
						sp.random.normal(init_params[0], init_params[1], Nn), 
						args = (Rr, Yy, precision), method='SLSQP')
	
	return res.x
	
def decode_nonlinear_CS(obj, opt_type="L1_strong", precision='None', 
				init_params=[sp.random.rand()*-0.1, sp.random.rand()*0.1]):
	"""
	Run CS decoding with L1 norm, using full activity with no linearization.
	
	Args:
		Rr: numpy array; measurement matrix.
		Yy: numpy array; Measured signal.
	
	Optional args:
		opt_type: String for type of L1 minimization "L1_strong" or "L1_weak".
		precision: Float, for L1_weak, the multiplier of the squared error.
		init_params: List; initialization point for the optimization.
	"""		

	def L1_strong(x):
		return sp.sum(abs(x))

	from kinetics import receptor_activity
	
	if opt_type == "L1_strong":
		constraints = ({'type': 'eq', 'fun': lambda x: 
						receptor_activity(x, obj.Kk1, obj.Kk2, obj.eps) - obj.Yy})
		res = minimize(L1_strong, 
						obj.Ss*sp.random.normal(1, 0.1, obj.Nn),
						#sp.random.normal(init_params[0], init_params[1], obj.Nn), 
						method='SLSQP', constraints = constraints)
	else:
		print ('Unknown optimization type')
		quit()
	
	return res.x