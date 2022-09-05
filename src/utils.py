"""
Utilities

Created by Nirag Kadakia at 15:12 06-27-2019
This work is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License.
To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import os
import shutil
import subprocess
import numpy as np
from load_save_data import load_settings, load_exp_matrix


class convert_units():
	"""
	Class for operations in converting between camera and projector units.

	Attrs
	-------

	Axx, Axy, Ayx, Ayy, Bx, By: floats:
		coefficients from affine transform from camera to stimulus units.
	Ainvxx, Ainvxy, Ainvyx, Ainvyy, Binvx, Binvy: floats:
		coefficients from affine transform from stimulus to camera units.
	"""

	def __init__(self, config=None):
		"""
		Constructor class for loading calibration coefficents.

		Args
		-------

		config: configuration object
			Holds configuration settings. If None, will read from settings.ini directly.
		"""

		if config is None:
			self.config = load_settings()
		else:
			self.config = config

		# Get projector calibration settings
		self.Axx = self.config.getfloat("Projector", "calibration_axx")
		self.Axy = self.config.getfloat("Projector", "calibration_axy")
		self.Ayx = self.config.getfloat("Projector", "calibration_ayx")
		self.Ayy = self.config.getfloat("Projector", "calibration_ayy")
		self.Bx = self.config.getfloat("Projector", "calibration_bx")
		self.By = self.config.getfloat("Projector", "calibration_by")

		self.Ainvxx = self.config.getfloat("Projector", "calibration_ainvxx")
		self.Ainvxy = self.config.getfloat("Projector", "calibration_ainvxy")
		self.Ainvyx = self.config.getfloat("Projector", "calibration_ainvyx")
		self.Ainvyy = self.config.getfloat("Projector", "calibration_ainvyy")
		self.Binvx = self.config.getfloat("Projector", "calibration_binvx")
		self.Binvy = self.config.getfloat("Projector", "calibration_binvy")

	def conv_proj_to_cam(self, stim_x, stim_y):
		"""
		Convert units from projector to camera

		Args
		-------

		stim_x, stim_y: floats
			location in projector units; between -1 and 1

		Returns
		-------

		detect_x, detect_y: floats or numpy array of floats
			location in camera pixel units corresponding to stim_x, stim_y
		"""

		detect_x = self.Ainvxx*stim_x + self.Ainvxy*stim_y + self.Binvx
		detect_y = self.Ainvyx*stim_x + self.Ainvyy*stim_y + self.Binvy

		return detect_x, detect_y

	def conv_cam_to_proj(self, detect_x, detect_y):
		"""
		Convert units from camera pixels to projector.

		Args
		-------

		detect_x, detect_y: floats
			location in camera pixel units corresponding to stim_x, stim_y

		Returns
		-------

		stim_x, stim_y: floats
			location in projector units; between -1 and 1
		"""

		stim_x = self.Axx*detect_x + self.Axy*detect_y + self.Bx
		stim_y = self.Ayx*detect_x + self.Ayy*detect_y + self.By

		return stim_x, stim_y


def overwrite_exp_matrix_keys(obj, key='x', write_override=False):
		"""
		Determine if exp_matrix exists, and if so, if a given key exists.

		Args
		-------

		obj: rec_and_proj instance
			Holds all experimental parameters
		key: key to check for in the matrix
		write_override: bool
			if True, does not ask user to overwrite

		Returns
		-------

		exp_matrix: dictionary or None
			Data for all attributes of all recorded flies at all frames.
			If None, exp_matrix has not yet been created.
		key_exists: bool
			if True, 'key' exists in the matrix.
		"""

		try:
			exp_matrix = load_exp_matrix(obj)
		except FileNotFoundError:
			return None, True
		try:
			exp_matrix[key]
		except KeyError:
			return exp_matrix, True
		if write_override == False:
			keys = input("Position and angle data exists in experimental "
				"matrix. Overwrite? (y/n): ")
			if keys == 'y':
				return exp_matrix, True
			else:
				return exp_matrix, False
		else:
			return exp_matrix, True


def get_consecutive_lists(data, stepsize=1):
	"""
	Get list of arrays, where each array is an unbroken sequence of consecutive
	numbers in an array.

	Args
	-------

	data: numpy array
	stepsize: int
		Step size of consecutive elements in the arrays

	Returns
	-------

	list:
		List of arrays.
	"""

	return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def get_non_nan_splits(arr):
	"""
	Get array of lists, where each array is the indices of an unbroken
	sequence of non-nan elements in a list.

	Args
	-------

	arr: numpy array
		Array to get non-nan lists from

	Returns
	-------

	list:
		List of arrays.
	"""

	return get_consecutive_lists(np.where(np.isfinite(arr))[0])


def get_traj_ranges(exp_matrix):
	"""
	Get array of lists, where each array is the indices of a single
	trajectory in the experimental matrix.

	Args
	-------

	exp_matrix: dict object
		Experimental data

	Returns
	-------

	traj_ranges: list
		Each element is a numpy array of indices of that trajectory.

	"""

	trjn = exp_matrix['trjn']
	idxs = np.arange(len(trjn))
	traj_ranges = np.split(idxs, np.where(np.diff(trjn) != 0)[0] + 1)
	missing_IDs = np.sort(list(set(range(trjn[-1])) - set(trjn)))
	for ID in missing_IDs:
		traj_ranges.insert(ID, [])

	return traj_ranges


def get_git_commit(path=None):
	"""
	Get active git branch name and git commit
	"""
	if path is None:
		path = os.path.dirname(os.path.realpath(__file__))
	
	branch = str(subprocess.check_output(['git', 'branch'], cwd=path, universal_newlines=True))
	branch = [a for a in branch.split('\n') if a.find('*') >= 0][0]
	branch = branch[branch.find('*') + 2 :]
	
	commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path).decode('ascii').strip()

	return branch, commit


def convert_dict_to_tuples(data_dirs):
	new_data_dirs = []
	for kk in data_dirs.keys():
		# iterate over keys and create tuples
		for data_path in data_dirs[kk]:
			new_data_dirs.append((kk, data_path))

	return new_data_dirs

def copy_file(src, dest):
	return shutil.copy2(src, dest)
