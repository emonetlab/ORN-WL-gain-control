"""
Functions for generating plot formats for various types of plots.

Created by Nirag Kadakia at 21:46 08-06-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
from local_methods import def_data_dir
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib.pyplot as plt

					
DATA_DIR = def_data_dir()
VAR_STRINGS = dict(mu_Ss0 = '$\langle s_0 \\rangle$', 
					mu_dSs = '$\langle \Delta s \\rangle$',
					mu_eps = '$\epsilon$',
					sigma_Ss0 = '$\Delta s_0$',
					sigma_dSs = '$\Delta s$',
					sigma_eps = '$\langle \epsilon^\mu -' 
						'\langle \epsilon \\rangle \\rangle$',
					A0 = '$a_0$',
					sigma_A0 = '$\sigma_{a_0}$', 
					mu_A0 = '$\mu_{a_0}$',
					receptor_tuning_center_mean = 
						'$ \langle a_\mu \rangle_\mu$',
					receptor_tuning_range_hi = 
						'$\max_{\mu} a_\sigma$',
					receptor_tuning_range_lo = 
						'$a_\sigma$',
					receptor_tuning_range_center_dev = 
						'$\langle \langle a_\mu \\rangle \\rangle_\mu$',
					seed_dSs = 'Signal ID')

def MSE_error_plots_formatting(x_axis_var):
	""" 
	Script to generate plots of errors versus inner loop variable, 
	for each outer variable.
	
	Args:
		x_axis_var: The inner loop variable, x-axis of each plot.
				
	Returns:
		fig: The figure object.
	"""
	
	fig = generic_plots()
	
	plt.yscale('log')
	plt.ylabel(r'MSE', fontsize = 20)
	try:
		plt.xlabel(r'%s' % VAR_STRINGS[x_axis_var], fontsize = 20)
	except:
		print ('No formatted x-label in dictionary, using generic x-axis '
					'label instead')
		plt.xlabel(r'$x$')
	
	return fig
	
def binary_error_plots_formatting(x_axis_var):
	""" 
	Script to generate plots of errors versus inner loop variable, 
	for each outer variable.
	
	Args:
		x_axis_var: The inner loop variable, x-axis of each plot.
				
	Returns:
		fig: The figure object.
	"""
	
	fig = generic_plots()
	
	plt.ylabel(r'Correct components (pct)', fontsize = 20)
	try:
		plt.xlabel(r'%s' % VAR_STRINGS[x_axis_var], fontsize = 20)
	except:
		print ('No formatted x-label in dictionary, using generic x-axis '
					'label instead')
		plt.xlabel(r'$x$')
	
	return fig	
	
def generic_plots():
	"""
	Generate generic plot format in reasonably pretty layout.
	
	Returns: 
		fig: The figure object.
	"""
	
	fig = plt.figure()
	fig.set_size_inches(3.5, 3.5)
	ax = plt.subplot(111)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	
	return fig

def temporal_plots():
	"""
	Generate generic temporal plot layout.
	
	Returns: 
		fig: The figure object.
	"""
	
	fig = plt.figure()
	fig.set_size_inches(8, 5)
	ax = plt.subplot(111)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlabel(r'Time (s)')
	
	return fig
