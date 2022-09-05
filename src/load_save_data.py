"""
Functions for loading and saving data from file.

Created by Nirag Kadakia at 13:52 06-17-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import os
import time
import datetime
import pickle
import gzip
import h5py
import configparser
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt


def get_data_dir():
	"""
	Get the local data directory from settings.ini
	
	Returns
	-------
	
	data_dir: str
		Absolute path of data directory
	"""	
	
	return load_settings()["Folders"]["data_dir"]


def get_analysis_dir():
	"""
	Get the local analysis directory from settings.ini
	
	Returns
	-------
	
	analysis_dir: str
		Absolute path of analysis directory
	"""	
	
	return load_settings()["Folders"]["analysis_dir"]


def get_rec_smoke_vids_dir():
	"""
	Get the directory in which videos of smoke are saved
	
	Returns
	-------
	
	data_dir: str
		Absolute path of data directory
	"""	
	
	return load_settings()["Folders"]["rec_smoke_vids_dir"]


def get_stim_vids_dir():
	"""
	Get the directory in which stimulus videos are saved
	
	Returns
	-------
	
	data_dir: str
		Absolute path of data directory
	"""	
	
	return load_settings()["Folders"]["stim_vids_dir"]


def smoke_assay_data_dir():
	"""
	Get the directory in data from the recorded smoke video are saved.
	
	Returns
	-------
	
	data_dir: str
		Absolute path of data directory
	"""	
	
	return load_settings()["Folders"]["smoke_assay_data_dir"]


def load_settings():
	"""
	Load the configuration file.
	
	Returns
	-------
	
	config: configparser.ConfigParser object
		Loaded configuration of the experiment.
	"""	
	
	config_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'settings.ini')

	config = configparser.ConfigParser()
	config.read(config_path)

	return config


def save_settings(config):
	"""
	Save updated configuration file.
	
	Args
	-------
	
	config: configparser.ConfigParser object
		Configuration of the experiment to save.
	"""
	
	config_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'settings.ini')

	with open(config_path, 'w') as config_file:
		config.write(config_file)


def set_exp_dir(exp_name):
	"""
	Set the folder for the experimental data to be saved in.
	
	Args
	-------
	
	exp_name: str
		Name of experiment; will be put in relevant dated folder.
	
	Returns
	-------
	
	out_dir: str
		Name of local directory to which data is being saved. 
	exp_dir: str
		Name of sub-directory within machine-dependent `data_dir` to 
		which data is being saved.
	"""
	
	currentDT = datetime.datetime.now()
	date = f'{currentDT.year}-{currentDT.month}-{currentDT.day}'
	exp_dir = os.path.join(date, f'{exp_name}_0')
	out_dir = os.path.join(get_data_dir(), exp_dir)
	iD = 0
	while os.path.exists(out_dir):
		iD += 1
		exp_dir = os.path.join(date, f'{exp_name}_{iD}')
		out_dir = os.path.join(get_data_dir(), exp_dir)
	os.makedirs(out_dir)
	
	return out_dir, exp_dir


def agg_shape_data(shapes):
	"""
	Aggregate the data from a list of PsychoPy ShapeStim objects at one 
	recorded frame into a separate list for each attribute.
	
	Args
	-------
	
	shapes: psychopy.elementArrayStim object 
		Shapes from which to extract vertices, position, size, colors, opacity.
	
	Returns
	-------
	
	shapes_to_save: dict
		Values are values of attribute (key) for elementArrayStim object
	"""
	
	# Unpack shapes from ShapeStim weakref to pickle
	shapes_to_save = dict()
	attrs_to_save = ['fieldPos', 'fieldSize', 'fieldShape', 'nElements', 
			'sizes', 'xys', 'rgbs', 'colors', 'colorSpace', 'depths', 
			'fieldDepth', 'oris', 'sfs', 'contrs', 'phases', 'texRes', 
			'interpolate', 'maskParams'] 
	
	for key in attrs_to_save:
		exec('shapes_to_save[key] = shapes.%s' % key)
	
	shapes_to_save['elementTex'] = shapes.tex
	shapes_to_save['elementMask'] = shapes.mask
	
	return shapes_to_save


def save_exp_data(obj):
	"""
	Save the rec_and_proj object, containing all projected shapes, stimuli
	parameters, fly locations, and camera, proje, and calibration parameters.
	
	Args
	-------
	
	obj: rec_and_proj instance
		Holds all attributes and data of experiment to be saved.
	"""
	
	out_dir = os.path.join(get_data_dir(), obj.exp_dir)
	
	# Unpack shapes from ShapeStim weakref to pickle
	out_file = os.path.join(out_dir, 'projected_shapes.pklz')
	with gzip.GzipFile(out_file, 'wb') as f:
		pickle.dump(obj.shapes_vec, f)
	
	out_file = os.path.join(out_dir, 'detected_flies.pklz')
	with gzip.GzipFile(out_file, 'wb') as f:
		pickle.dump({'pos': obj.fly_pos_vec, 'states': obj.fly_states_vec}, f)
	
	# For video stimuli, delete the video frames. Delete all psychopy weakrefs
	try:
		del obj.stim.win
	except:
		pass
	try:
		del obj.stim.shapes
	except:
		pass
	try:
		del obj.stim.vid_frames
	except:
		pass
	
	# Delete all daq weakrefs
	try:
		del obj.odor_wind.daq
	except:
		pass
	if obj.bypass_cam == False:
		del obj.c
	else:
		del obj.cap
	del obj.detector
	del obj.shapes_vec
	del obj.video
	del obj.centroids
	del obj.states
	
	out_file = os.path.join(out_dir, 'obj.pklz')
	with gzip.GzipFile(out_file, 'wb') as f:
		pickle.dump(obj, f)


def load_exp_data(dir=None, subdir=None, exp_name=None, load_obj=True, 
				  load_flies=True, load_shapes=True):
	"""
	Load raw data recorded from experiment: obj, flies, and projected shapes.
	Can either pass the full directory of the recorded data (dir) or
	the subdir (by date) and the experimental folder name (exp_name).
	Either `dir' or both `subdir' and `exp_name' must be passed. The former
	option is a full absolute path; the latter is just the name of the 
	subdir and exp_name; the location of the rec_videos is taken from 
	get_data_dir().
	
	Args
	-------
	
	dir: string
		Full path of dataset, has form data_dir/subdir/exp_name, where
		subdir is the date of the experiment and exp_name is its name
	subdir: string
		Experiment date (ie folder name)
	exp_name: string
		Folder of experiment within subdir
			
	Returns
	-------
	
	obj: rec_and_proj
		Holds all experimental parameters
	flies: dict
		Holds position and states (present: 1; absent: 0) of all flies, 
		keyed by fly number, at all frames.
	shapes: dict
		Values are values of attribute (key) for elementArrayStim object
	
	"""
	# dir is a built-in function, consider changing the variable name
	if dir is not None:
		pass
	else: 
		assert (subdir is not None) and (exp_name is not None), \
		  ("Can either pass the full directory of the experimental data (dir) "
		   "or the subdir (by date) and the experimental folder name "
		   "(exp_name). Either `dir' or both `subdir' and `exp_name' "
		   "must be passed")
		dir  = os.path.join(get_data_dir(), subdir, exp_name)
	obj_file = os.path.join(dir, 'obj.pklz')
	
	if load_obj == True:
		with gzip.open(obj_file, 'rb') as f:
			obj = pickle.load(f)

			# path is hardcoded for windows when saved
			# convert it to current operating system
			exp_dir = obj.exp_dir.split('\\')
			obj.exp_dir = os.path.join(exp_dir[0], exp_dir[1])
	else:
		obj = None

	flies_file = os.path.join(dir, 'detected_flies.pklz')
	if load_flies == True:
		with gzip.open(flies_file, 'rb') as f:
			flies = pickle.load(f)
	else:
		flies = None
	
	shapes_file = os.path.join(dir, 'projected_shapes.pklz')
	if load_shapes == True:
		with gzip.open(shapes_file, 'rb') as f:
			shapes = pickle.load(f)

	else:
		shapes = None
	return obj, flies, shapes


def save_exp_matrix(exp_matrix, obj):
	"""
	Save experimental matrix
	
	Args
	-------
	
	exp_matrix: dictionary
		Data for all attributes of all recorded flies at all frames
	obj: rec_and_proj
		Holds all experimental parameters
	"""
	
	filename = os.path.join(get_data_dir(), obj.exp_dir, 'exp_matrix.pklz')
	with gzip.GzipFile(filename, 'wb') as f:
		pickle.dump(exp_matrix, f)


def load_exp_matrix(obj):
	"""
	Save experimental matrix
	
	Args
	-------
	
	obj: rec_and_proj obj
		Holds all experimental parameters
	
	Returns
	-------
	
	exp_matrix: dictionary
		Data for all attributes of all recorded flies at all frames
	"""
	
	filename = os.path.join(get_data_dir(), obj.exp_dir, "exp_matrix.pklz")
	with gzip.open(filename, 'rb') as f:
		exp_matrix = pickle.load(f)
		
	return exp_matrix


def print_all_exp_in_subdir(subdir):
	"""
	Save experimental matrix
	
	Args
	-------
	
	subdir: string
		Experiment date (ie folder name)
	
	"""
	
	print('Ordering of experiments is machine-dependent. Do not trust.\n')
	
	subdir_path = os.path.join(get_data_dir(), subdir)
	exp_names = next(os.walk(subdir_path))[1]
	
	stim_attrs = dict()
	stim_attrs['horiz_moving_bars'] = ['speed', 'on_width', 'off_width']
	stim_attrs['vert_moving_bars'] = ['speed', 'on_width', 'off_width']
	stim_attrs['full_field_flash'] = ['freq', 'dur', 'On_T', 'Off_T'] 
	stim_attrs['scintillator_2'] = ['motion_ori', 'direction', 'corr_sign', 
									'px_per_bar', 't_step', 'corr_x_step' 
									'switch_T', 'motion_T']
	stim_attrs['scintillator_2'] = ['motion_ori', 'direction', 'corr_sign', 
									'px_per_bar', 't_step', 'corr_x_step', 
									'switch_T', 'motion_T', 'num_corrs', 
									'static_motion_off']
	stim_attrs['noisy_glider'] = ['motion_ori', 'direction', 'corr_sign', 
								  'px_per_bar', 't_step', 'corr_x_step', 
								  'corr_x_step_blank', 'switch_T',  
								  'motion_T', 'static_motion_off']
								  
	for exp_name in exp_names:
		obj, _, _ = load_exp_data(subdir=subdir, exp_name=exp_name, 
								  load_flies=False, load_shapes=False)
		stim_protocol_name = obj.stim_protocol.__name__
		print(exp_name)
		print('  ', stim_protocol_name)
		if stim_protocol_name in stim_attrs.keys():
			for key in stim_attrs[stim_protocol_name]:
				exec('print("     ", key, ":", obj.stim.%s)'  % (key))
		print('\n')


def load_rec_vid_frame(subdir, exp_name, frame):
	"""
	Load raw video frame from experiment.
	
	Args
	-------
	
	subdir: string
		Experiment date (ie folder name)
	exp_name: string
		Folder of experiment within subdir
	frame: int
		Frame number to load
		
	Returns
	-------
	
	cam_frame: (N, M, 3) numpy array
		8-bit RGB values of each pixel in frame
	"""
	
	import cv2
	vid_path = os.path.join(get_data_dir(), subdir, exp_name, 'frames.avi')
	cap = cv2.VideoCapture(vid_path)
	cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
	_, cam_frame = cap.read()
	
	return cam_frame
	
def load_opto_stim_frame(subdir, exp_name, frame, 
					   conv_to_cam_units=True, obj=None, 
					   stim_subsample=4):
	"""
	Load projected stimulus video frame from saved stim.mp4. Note that
	this saved stim file may be saved at a lower temporal resolution than 
	the recording to save on space -- so it is not a faithful representation
	of the stimulus delivered, but is just used for plotting and comparison.
	the 
	
	Args
	-------
	
	subdir: string
		Experiment date (ie folder name)
	exp_name: string
		Folder of experiment within subdir
	frame: int
		Frame number to load
	conv_to_cam_units: bool
		Whether to convert to same units as camera pixels
	obj: rec_and_proj obj
		Holds all experimental parameters. Only needed if conv_to_cam_units
		is True.
	stim_subsample: int
		Stimulus is saved every nth frame to save on space; typically this is
		set to 4 frames every camera frame.
		
	Returns
	-------
	
	stim_frame: (N, M, 3) numpy array
		8-bit RGB values of each pixel of stimulus in that frame
	"""
	
	import cv2
	stim_path = os.path.join(get_data_dir(), subdir, exp_name, 'stim.mp4')
	cap = cv2.VideoCapture(stim_path)
	cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame/stim_subsample))
	_, stim_frame = cap.read()
	
	# Psychopy stimuli are in BGR
	stim_frame = stim_frame[:, :, ::-1]

	if conv_to_cam_units is True:
		
		ainvxx = obj.config.getfloat("Projector", "calibration_ainvxx")
		ainvxy = obj.config.getfloat("Projector", "calibration_ainvxy")
		ainvyx = obj.config.getfloat("Projector", "calibration_ainvyx")
		ainvyy = obj.config.getfloat("Projector", "calibration_ainvyy")
		binvx = obj.config.getfloat("Projector", "calibration_binvx")
		binvy = obj.config.getfloat("Projector", "calibration_binvy")

		proj_x_res = obj.config.getfloat("Projector", "res_dx")
		proj_y_res = obj.config.getfloat("Projector", "res_dy")
		cam_x_res = int(obj.config.getfloat("Camera", "dx"))
		cam_y_res = int(obj.config.getfloat("Camera", "dy"))

		M = [[ 2*ainvxx/proj_x_res, 
			  -2*ainvxy/proj_y_res, 
			  -ainvxx + ainvxy + binvx], 
			 [ 2*ainvyx/proj_x_res, 
			  -2*ainvyy/proj_y_res, 
			  -ainvyx + ainvyy + binvy]] 
		M = np.array(M)
		
		stim_frame = cv2.warpAffine(stim_frame, M, (cam_x_res, cam_y_res))
		
	return stim_frame


def save_metadata(metadata, obj):
	"""
	Save metadata of analysis
	
	Args
	-------
	
	metadata: dictionary
		Data for all metadata of the analysis
	obj: rec_and_proj
		Holds all experimental parameters
	"""
	
	filename = os.path.join(get_data_dir(), obj.exp_dir, 'metadata.pkl')
	with gzip.GzipFile(filename, 'wb') as f:
		pickle.dump(metadata, f)
		

def load_metadata(obj):
	"""
	Load metadata of analysis
	
	Args
	-------
	
	obj: rec_and_proj
		Holds all experimental parameters
	
	Returns
	-------
	
	metadata: dictionary
		Data for all metadata of the analysis
	"""
	
	filename = os.path.join(get_data_dir(), obj.exp_dir, 'metadata.pkl')
	with gzip.open(filename, 'rb') as f:
		metadata = pickle.load(f)
		
	return metadata


def gen_plot(width, height, dpi=75):
	"""
	Generic plot for all figures to get right linewidth, font sizes, 
	and bounding boxes
	"""
	
	fig = plt.figure(figsize=(width, height), dpi=dpi)
	ax = plt.subplot(111)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(direction='in', length=3, width=0.5)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(0.5)
	
	return fig, ax


def save_fig(fig_name, subdir=None, clear_plot=True, tight_layout=True,	 
			 save_svg=True):
	"""
	Save figs in analysis folder.
	"""
	
	analysis_dir = get_analysis_dir()
	
	# Save locally
	if subdir is None:
		out_dir = os.join.path(analysis_dir, 'figures')
	else:
		out_dir = os.join.path(analysis_dir, 'figures', subdir)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if tight_layout == True:
		plt.tight_layout()
	filename = os.join.path(out_dir, fig_name)
	plt.savefig(f'{filename}.png', bbox_inches = 'tight')
	if save_svg == True:
		plt.savefig(f'{filename}.svg', bbox_inches = 'tight')


def load_smoke_assay_exp_matrix_pkl(subdir, file, assay_type):
	"""
	Load pickled experimental matrix file.
	
	Args
	-------
	
	subdir: string of directory within mahmut_demir/analysis
	file: string of encounter dataset file (-.mat)
	assay_type: structure of .mat to load.
		
	Returns
	-------
	
	exp_matrix_dict: Ordered dictionary whose keys are the encounter 
		events, e.g. 'trjn' = trajectory number; 'sx' = signal x-position.
		The dictionary values are arrays whose values are the value of 
		that key at all framenumbers (combined over all trajectories 
		and videos)
	"""
	
	time_start = time.time()
	print('Loading smoke data from pkl obj...')
	
	out_dir = os.path.join(smoke_assay_data_dir(), 'objects', subdir)
	if assay_type is None:
		pkl_file = os.path.join(out_dir, f'{file}.pkl')
	else:
		pkl_file = os.path.join(out_dir, file, f'{assay_type}.pkl')
	
	with open(pkl_file, 'rb') as f:
		exp_matrix_dict = pickle.load(f, encoding='latin1')
	
	print('Loaded pkl file in %.3f seconds' % (time.time() - time_start))
	
	return exp_matrix_dict


def load_vid_by_frm(subdir='2018_05_07 Intermitten Smoke - No Flies',
					file='2018_05_07_NoFly_3_0ds_0do_ISV900ul5ll01_1', 
					frame=None, bck_sub=True):
	"""
	Load a single frame of a flywalk movie from the h5 file. 

	Args
	-------
	
	subdir: string of directory within mahmut_demir/analysis file: 
		string of encounter dataset file (-.mat) frame: int; frame to load
		bck_sub: bool; if True, subtract background

	Returns
	-------
	
	frm_data: (n, m) array; data for frame. 
	"""

	data_dir = os.path.join(get_rec_smoke_vids_dir(), subdir)
	vid_file = os.path.join(data_dir, f'{file}-frames.mat')
	bck_file = os.path.join(data_dir, f'{file}.mat')

	assert os.path.isfile(vid_file) == True, "%s not found" % vid_file
	
	with h5py.File(vid_file, 'r') as f:
		data_set = f['frames']
		frm_data = data_set[frame]

	if bck_sub == True:
		assert os.path.isfile(bck_file) == True, "%s not found" % bck_file
		with h5py.File(bck_file, 'r') as f:
			bck_data = f['p']['bkg_img'][:]
			frm_data = np.maximum(frm_data.astype(np.float32) - 
									bck_data.astype(np.float32), 
									np.zeros(frm_data.shape)).astype(np.int8)
		
	return frm_data


def parse_mat_dict(data):
	"""
	Parse a matlab structure of newer or older version into a dictionary.

	Args
	-------
	
	data: matlab data file loaded by either spio.loadmat() with 
		struct_as_record=False and squeeze_me=True, or by
		h5py.File().

	Returns
	-------
	
	mat_dict: python dictionary from matlab data structure
	"""

	def _check_keys(dict):
		"""
		Checks if entries in dictionary are mat-objects. If yes
		todict is called to change them to nested dictionaries
		"""

		for key in dict:
			if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
				dict[key] = _todict(dict[key])
		return dict		   

	def _todict(matobj):
		"""
		A recursive function which constructs from matobjects nested dictionaries
		"""

		dict = {}
		for strg in matobj._fieldnames:
			elem = matobj.__dict__[strg]
			if isinstance(elem, spio.matlab.mio5_params.mat_struct):
				dict[strg] = _todict(elem)
			else:
				dict[strg] = elem

		return dict

	mat_dict = _check_keys(data)

	return mat_dict


def load_metadata_mat(subdir='2018_05_07 Intermitten Smoke - No Flies',
					file='2018_05_07_NoFly_3_0ds_0do_ISV900ul5ll01_1'):

	"""
	Load the metadata corresponding to a recorded smoke video file 

	Args
	-------
	
	subdir: string of directory within mahmut_demir/analysis
		file: string of encounter dataset file (-.mat)

	Returns
	-------
	
	mat_dict: dict
		Contains the metadata
	"""

	mat_file = os.path.join(get_rec_smoke_vids_dir(), subdir, f'{file}.mat')
	assert os.path.isfile(mat_file) == True, "%s not found" % mat_file
	try:
		mat_data = h5py.File(mat_file, 'r')
	except (NotImplementedError, IOError):
		print('\nHDF5 not supported, trying scipy.io.loadmatos.path.dirname(__file__) +  /...')
		try:
			mat_data = spio.loadmat(mat_file, struct_as_record=False, 
						squeeze_me=True)
		except (NotImplementedError, IOError):
			print('File %s of non-supported type!' % mat_file)

	# Parse matlab structure; read experimental matrix; transpose if needed
	mat_dict = parse_mat_dict(mat_data)

	return mat_dict


def save_stim_video_metadata(metadata, stim_name):
	"""
	Save the metadata for a stimulus video file.

	Args
	-------
	
	metadata: dict
		Contains the metadata to be saved.
	stim_name: str
		Name of the stimulus video file.
	"""

	filename = os.path.join(get_stim_vids_dir(), f'{stim_name}_metadata.pkl')
	pickle.dump(metadata, open(filename, 'wb'))


def load_stim_video_metadata(stim_name):
	"""
	Load the metadata for a stimulus video file.

	Args
	-------
	
	stim_name: str
		Name of the stimulus video file.
	
	
	Returns
	-------
	
	metadata: dict
		Contains the metadata to be saved.
	"""
	
	filename = os.path.join(get_stim_vids_dir(), f'{stim_name}_metadata.pkl')
	metadata = pickle.load(open(filename, 'rb'))
	
	return metadata