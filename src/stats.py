"""
General, miscellaneous functions for statistical analysis of data.

Created by Nirag Kadakia at 18:03 08-11-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import rv_continuous

def power_law_regress(x, y):
	"""
	Plot a power law fit on an existing figure
	"""
	slope, y_int, r_value, p_value, std_err = linregress(sp.log(x), sp.log(y))
	plt.plot(x, sp.exp(slope*sp.log(x) + y_int), color = 'orangered', 
				linestyle='--', linewidth = 3)
	
	print('Power Law: slope = %.5f...y_int = %.5f, r_value = %.5e...p_value = '\
			'%.5e...std_err = %.5e' % (slope, y_int, r_value, p_value, std_err))
	
	return slope, y_int, r_value, p_value, std_err

def lognormal_regress(x, y):
	"""
	Plot a lognormal fit on an existing figure
	"""

	slope, y_int, r_value, p_value, std_err = linregress(sp.log(x), y)
	plt.plot(x, slope*sp.log(x) + y_int, color = 'orangered', 
				linestyle='--', linewidth = 3)

	print('Lognormal: slope = %.5f...y_int = %.5f, r_value = %.5e...p_value ='
			'%.5e...std_err = %.5e' % (slope, y_int, r_value, p_value, std_err))
	
	return slope, y_int, r_value, p_value, std_err

class Kk_dist_Gaussian_activity(rv_continuous):
	"""
	Random variable of Kk2 = K2 (activated disassociation constants), 
	given a normally-distributed activity level for a given stimulus, 
	epsilon, and presumed receptor-dependent activity statistics 
	(mean and sigma)
	"""

	def _argcheck(self, *args):
		# Override argument checking
		return 1

	def _pdf(self, Kk, activity_mu, activity_sigma, Ss0, eps):
		C = sp.exp(-eps)*Ss0
		prefactor = C*(2*sp.pi*activity_sigma**2.0)**.5
		exp_arg = (activity_mu - 1./(Kk/C + 1))/(2*activity_sigma**2.0)**.5
		
		return 1/prefactor*sp.exp(-exp_arg**2.0)/(Kk/C + 1)**2.0	
		
class A0_dist_norm_Kk(rv_continuous):
	"""
	Distribution of activity levels given normally-distributed
	Kk2 = K2 disassociation constants at activation.
	"""

	def _argcheck(self, *args):
		# Override argument checking
		return 1

	def _pdf(self, A0, Ss0, eps, mu_Kk2, sigma_Kk2):
		C = sp.exp(-eps)*Ss0
		prefactor = C/(2*sp.pi*sigma_Kk2**2.0)**.5
		exp_arg = (mu_Kk2 + C - C/A0)/(2*sigma_Kk2**2.0)**.5
		
		return prefactor*sp.exp(-exp_arg**2.0)/(A0)**2.0
		