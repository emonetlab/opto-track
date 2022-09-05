"""
Generate fly variables (position, speed, orientation, etc.), 
and stimulus data arrays from file

Created by Nirag Kadakia at 21:00 08-13-2019
This work fly_num licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
import time
import os
import sys
import cv2
from glob import glob
from skimage.feature import canny
from scipy.signal import savgol_filter
from track_funcs import fit_ellipse, opt_ellipse_orientation
from load_save_data import load_exp_data, load_exp_matrix, save_exp_matrix, \
						   save_metadata, load_metadata, get_data_dir
from utils import overwrite_exp_matrix_keys, get_traj_ranges


class gen_xy_theta(object):
	"""
	Class for generating the x-y and theta data for an entire directory 
	by fitting an ellipse to fly positions, then correcting for heading.
	
	Time is saved in seconds, position in mm, and angle in degrees modded
	between 0 and 360.
	"""
	
	def __init__(self, subdir='2019-8-3', crop_size_mm=2, 
				 ellipse_detect_threshs=[3, 50], ellipse_detect_sigma=3,
				 min_walk_spds=np.arange(1, 4.1, 0.25), 
				 v_smooth_windows=np.arange(51, 152, 50), 
				 min_walk_dur=0.3):
		
		self.exp_dirs = glob("%s/%s/*/" % (get_data_dir(), subdir))
		self.crop_size_mm = crop_size_mm
		self.ellipse_detect_threshs = ellipse_detect_threshs
		self.ellipse_detect_sigma = ellipse_detect_sigma
		self.min_walk_spds = min_walk_spds
		self.v_smooth_windows = v_smooth_windows
		self.min_walk_dur = min_walk_dur
		
	def gen_data_all_exp_dirs(self, overwrite_all=False):
		"""
		Generate x, y, and orientation from all experimental directories 
		in subdir.
		"""
		
		for exp_dir in self.exp_dirs:
			self.gen_data_single_exp_dir(exp_dir, write_override=overwrite_all)
				
	def gen_data_single_exp_dir(self, exp_dir, write_override=False):
		"""
		Gen data for single experimental directory.
		"""
		
		obj, flies, _ = load_exp_data(exp_dir)
		xy_pos = flies['pos']
		states = np.asarray(flies['states'])
		
		# Fly IDs which are active at least one frame
		num_active_flies = np.sum(np.prod(1 - states, axis=0) == 0)
		if num_active_flies == 0:
			return None
			
		# To hold data for all trajectories
		trjn = []
		t = []
		x = []
		y = []
		theta = []
		theta_uncut = []
		
		# Check if keys exist, prompt to overwrite, if so
		exp_matrix, overwrite = overwrite_exp_matrix_keys(obj, key='x', 
								  write_override=write_override)
		if not overwrite:
			print('Not overwriting %s..' % exp_dir) 
			return None
		if exp_matrix is None:
			exp_matrix = dict()
			metadata = dict()
		else:
			metadata = load_metadata(obj)
				
		for fly_num in range(num_active_flies):
			
			_t, _trjn, _x, _y, _theta, _theta_uncut = \
				self.gen_data_single_traj(exp_dir, fly_num, xy_pos, 
										  states, obj)
			
			t.extend(_t)
			trjn.extend(_trjn)
			x.extend(_x)
			y.extend(_y)
			theta.extend(_theta)
			theta_uncut.extend(_theta_uncut)
		
		exp_matrix['t'] = np.array(t)
		exp_matrix['trjn'] = np.array(trjn)
		exp_matrix['x'] = np.array(x)
		exp_matrix['y'] = np.array(y)
		exp_matrix['theta'] = np.array(theta)
		exp_matrix['theta_uncut'] = np.array(theta_uncut)
		
		metadata['theta_calc'] = dict()
		metadata['theta_calc']['ellipse_detect_threshs'] = \
			self.ellipse_detect_threshs
		metadata['theta_calc']['ellipse_detect_sigma'] = \
			self.ellipse_detect_sigma
		
		save_exp_matrix(exp_matrix, obj)
		save_metadata(metadata, obj)
		
		return None
		
	def gen_data_single_traj(self, exp_dir, fly_num, xy_pos, states, obj):
		"""
		X-y positions and theta for a single trajectory.
		Positions updated by fitting ellipse to fly, then getting center 
		position; this is more accurate than blob detection while tracking. 
		Orientations are flip-corrected using the fly heading.
		"""

		# 1/2 side size of cropped image around fly in pixels 
		crop_im_sz = int(self.crop_size_mm/obj.mm_per_px)
		
		t = []
		trjn = []
		x = []
		y = []
		theta = []
		
		frms_to_read = np.where(states[:, fly_num])[0]
		if len(frms_to_read) == 0:
			return t, trjn, x, y, theta, theta
		
		out_file = os.path.join(exp_dir, f'orient_{fly_num}.avi')
		video = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'XVID'), 
								obj.rec_rate, (2*crop_im_sz, 2*crop_im_sz), 1)
		cap = cv2.VideoCapture(os.path.join(exp_dir, 'frames.avi'))
		cap.set(cv2.CAP_PROP_POS_FRAMES, frms_to_read[0])
				
		for frm_num in frms_to_read:
			print(exp_dir, fly_num, frm_num)
			ret, rec_img = cap.read()
			
			# This should not be but might happen on last frame of video
			# for some reason. Just replace with an array of nans. This 
			# could also be the caase if using dummy images for testing 
			# (i.e. running with the bypass_cam=True)
			if rec_img is None:
				try:
					blank_img
				except:
					blank_img = np.empty((10000, 10000, 3))
					blank_img[:] = np.nan
				rec_img = blank_img.astype('uint8')
			rec_img = np.swapaxes(rec_img, 0, 1)
				
			# Nominal center of ellipse, tracked from video live-time
			nom_center = xy_pos[frm_num][fly_num].astype(int)
			
			# Crop to around center, detect edges of ellipse
			min_x = max(nom_center[0] - crop_im_sz, 0)
			max_x = min(nom_center[0] + crop_im_sz, int(obj.cam_dx))
			min_y = max(nom_center[1] - crop_im_sz, 0)
			max_y = min(nom_center[1] + crop_im_sz, int(obj.cam_dy))
			crop_im = rec_img[min_x: max_x, min_y: max_y]
			imgray = cv2.cvtColor(crop_im, cv2.COLOR_BGR2GRAY)
			
			# Fit ellipse points to parameters. If cannot fit, append nan
			edges = canny(imgray, sigma=self.ellipse_detect_sigma, 
						  low_threshold=self.ellipse_detect_threshs[0], 
						  high_threshold=self.ellipse_detect_threshs[1])
			lsqe = fit_ellipse()
			try:
				lsqe.fit([np.where(edges)[0], np.where(edges)[1]])
				center, width, height, phi = lsqe.parameters()
				crop_im_T = np.swapaxes(crop_im, 0, 1)
				cv2.ellipse(crop_im_T, (int(center[0]), int(center[1])),
							(int(width), int(height)),
							phi, 0, 360, (255, 255, 0), -1)
			except (np.linalg.linalg.LinAlgError, IndexError, TypeError, 
			  ValueError):
				crop_im_T = np.swapaxes(crop_im, 0, 1)
				center = [np.nan, np.nan]
				phi = np.nan
			if isinstance(phi, complex):
				print('complex', phi)
				phi = np.nan
				
			# Update position, orientation for this fly
			t.append(frm_num/obj.rec_rate)
			trjn.append(fly_num)
			x.append((center[0] + min_x)*obj.mm_per_px)
			y.append((center[1] + min_y)*obj.mm_per_px)
			theta.append(phi % 360)
			
			# This conversion to BGR avoids some weird thing with opencv 
			# that crashes script on video.write without raising an error (?)
			video.write(cv2.cvtColor(crop_im_T,cv2.COLOR_RGB2BGR))
			
		video.release()
		
		# Get correct orientaiton of fly (distinguishign back to front), 
		# by optimizing fix_ellipse_orientation() over various smoothing 
		# and thresholding parameters
		theta, theta_uncut = opt_ellipse_orientation(np.array(theta), x, y, 
								self.min_walk_spds, obj.rec_rate, 
								self.v_smooth_windows,
								min_walk_dur=self.min_walk_dur)
		return t, trjn, x, y, theta, theta_uncut
		
		
class smooth_data(object):
	"""
	Class for generating smoothed position, orientation, heading, speed,
	stimulus, etc.
	"""
	
	def __init__(self, subdir='2019-8-3'):
		"""
		Set the list of directories
		"""
		
		self.exp_dirs = glob("%s/%s/*/" % (get_data_dir(), subdir))
		
	def smooth_xy_all_exp_dirs(self, window=21, order=4):
		"""
		Get xy, velocity smoothed data for all directories in subdir
		"""
		
		for exp_dir in self.exp_dirs:
			self.smooth_xy_single_exp_dir(exp_dir, window, order)
	
	def smooth_theta_all_exp_dirs(self, window=21, order=4):
		"""
		Get theta, dheta/dt smoothed data for all directories in subdir.
		"""
		
		for exp_dir in self.exp_dirs:
			self.smooth_theta_single_exp_dir(exp_dir, window, order)
	
	def smooth_signal_all_exp_dirs(self, window=21, order=4):
		"""
		Get signal smoothed data for all directories in subdir.
		"""
		
		for exp_dir in self.exp_dirs:
			self.smooth_signal_single_exp_dir(exp_dir, window, order)
		
	def smooth_xy_single_exp_dir(self, exp_dir, window=21, order=4, 
								 spd_jump_thresh=100, spd_jump_mask_T=0.5):
		"""
		Smooth position and velocity for single directory.
		"""
		
		obj, _, _ = load_exp_data(exp_dir)
		exp_matrix = load_exp_matrix(obj)
		metadata = load_metadata(obj)
	
		x = exp_matrix['x']
		y = exp_matrix['y']
		exp_matrix['x_smooth'] = np.empty(len(x))*np.nan
		exp_matrix['y_smooth'] = np.empty(len(x))*np.nan
		exp_matrix['vx_smooth'] = np.empty(len(x))*np.nan
		exp_matrix['vy_smooth'] = np.empty(len(x))*np.nan
		exp_matrix['jumps'] = np.zeros(len(x))
		
		dt = 1./obj.rec_rate
		
		for rng in get_traj_ranges(exp_matrix):
			if len(rng) < window:
				continue
			try:
				exp_matrix['x_smooth'][rng] = \
				  savgol_filter(x[rng], window, order, deriv=0)
				exp_matrix['y_smooth'][rng] = \
				  savgol_filter(y[rng], window, order, deriv=0)
				exp_matrix['vx_smooth'][rng] = \
				  savgol_filter(x[rng], window, order, deriv=1, delta=dt)
				exp_matrix['vy_smooth'][rng] = \
				  savgol_filter(y[rng], window, order, deriv=1, delta=dt)
			except np.linalg.LinAlgError:
				continue
		exp_matrix['spd_smooth'] = (exp_matrix['vx_smooth']**2 + 
									exp_matrix['vy_smooth']**2)**0.5
		
		# Remove jumps by thresholding speed
		mask_frms = int(obj.rec_rate*spd_jump_mask_T)
		for rng in get_traj_ranges(exp_matrix):
			jump_pts = np.where(exp_matrix['spd_smooth'][rng] 
							   > spd_jump_thresh)[0]
			for iT in jump_pts:
				beg = max(0, iT - mask_frms)
				end = min(len(rng), iT + mask_frms)
				mask_rng = rng[0] + np.arange(beg, end)
				exp_matrix['jumps'][mask_rng] = 1
		exp_matrix['spd_smooth'][np.where(exp_matrix['jumps'])] = np.nan
		exp_matrix['vx_smooth'][np.where(exp_matrix['jumps'])] = np.nan
		exp_matrix['vy_smooth'][np.where(exp_matrix['jumps'])] = np.nan
		
		metadata['xy_smooth_calc'] = dict()
		metadata['xy_smooth_calc']['spd_jump_thresh'] = spd_jump_thresh
		metadata['xy_smooth_calc']['spd_jump_mask_T'] = spd_jump_mask_T
		metadata['xy_smooth_calc']['smooth_window_len'] = window
		metadata['xy_smooth_calc']['polyorder'] = order
		
		save_exp_matrix(exp_matrix, obj)
		save_metadata(metadata, obj)
				
	def smooth_theta_single_exp_dir(self, exp_dir, window=21, order=4, 
									turn_jump_thresh=500, 
									turn_jump_mask_T=0.25):
		"""
		Get theta, dheta/dt smoothed data for single directory.
		"""
		
		obj, _, _ = load_exp_data(exp_dir)
		exp_matrix = load_exp_matrix(obj)
		metadata = load_metadata(obj)
		
		theta = exp_matrix['theta']
		theta_uncut = exp_matrix['theta_uncut']
		exp_matrix['theta_smooth'] = np.empty(len(theta))*np.nan
		exp_matrix['dtheta_smooth'] = np.zeros(len(theta))*np.nan
		
		dt = 1./obj.rec_rate
		for rng in get_traj_ranges(exp_matrix):
			if len(rng) < window:
				continue
			try:
				exp_matrix['theta_smooth'][rng] = \
				  savgol_filter(theta_uncut[rng], window, order, deriv=0) % 360
				exp_matrix['dtheta_smooth'][rng] = \
				  savgol_filter(theta_uncut[rng], window, order, deriv=1, 
				  delta=dt)
			except np.linalg.LinAlgError:
				continue
			
			# Just nan out the boundaries to prevent false positives
			ends = np.hstack((range(rng[0], rng[0] + window//2), 
							  range(rng[-1] - window//2, rng[-1])))
			exp_matrix['theta_smooth'][ends] = np.nan
			exp_matrix['dtheta_smooth'][ends] = np.nan
		
		_dtheta = exp_matrix['dtheta_smooth'].copy()
		
		# These are turn blips from bad orientation fitting or otherwise
		mask_frms = int(obj.rec_rate*turn_jump_mask_T)
		for rng in get_traj_ranges(exp_matrix):
			jump_pts = np.where(abs(_dtheta[rng]) > turn_jump_thresh)[0]
			for iT in jump_pts:
				beg = max(0, iT - mask_frms)
				end = min(len(rng), iT + mask_frms)
				mask_rng = rng[0] + np.arange(beg, end)
				exp_matrix['dtheta_smooth'][mask_rng] = np.nan
				
		# Nan dtheta at jumps as well
		exp_matrix['theta_smooth'][np.where(exp_matrix['jumps'])] = np.nan
		exp_matrix['dtheta_smooth'][np.where(exp_matrix['jumps'])] = np.nan
			
		metadata['theta_smooth_calc'] = dict()
		metadata['theta_smooth_calc']['turn_jump_thresh'] = turn_jump_thresh
		metadata['theta_smooth_calc']['turn_jump_mask_T'] = turn_jump_mask_T
		metadata['theta_smooth_calc']['smooth_window_len'] = window
		metadata['theta_smooth_calc']['polyorder'] = order
		
		save_exp_matrix(exp_matrix, obj)
		save_metadata(metadata, obj)

	def smooth_signal_single_exp_dir(self, exp_dir, window=101, order=4):
		"""
		Get signal smoothed data for single directory in subdir.
		"""
	
		obj, _, _ = load_exp_data(exp_dir)
		exp_matrix = load_exp_matrix(obj)
		metadata = load_metadata(obj)
		
		signal = exp_matrix['signal']
		exp_matrix['signal_R_smooth'] = np.empty(len(signal))*np.nan
		exp_matrix['signal_G_smooth'] = np.empty(len(signal))*np.nan
		exp_matrix['signal_B_smooth'] = np.empty(len(signal))*np.nan
		
		dt = 1./obj.rec_rate
		for rng in get_traj_ranges(exp_matrix):
			if len(rng) < window:
				continue
			try:
				exp_matrix['signal_R_smooth'][rng] = \
				  savgol_filter(signal[rng][:, 0], window, order, deriv=0)
				exp_matrix['signal_G_smooth'][rng] = \
				  savgol_filter(signal[rng][:, 1], window, order, deriv=0)
				exp_matrix['signal_B_smooth'][rng] = \
				  savgol_filter(signal[rng][:, 2], window, order, deriv=0)
			except np.linalg.LinAlgError:
				continue
			
			# Just nan out the boundaries to prevent false positives
			ends = np.hstack((range(rng[0], rng[0] + window//2), 
							  range(rng[-1] - window//2, rng[-1])))
			exp_matrix['signal_R_smooth'][ends] = np.nan
			exp_matrix['signal_G_smooth'][ends] = np.nan
			exp_matrix['signal_B_smooth'][ends] = np.nan
			
		metadata['signal_smooth_calc'] = dict()
		metadata['signal_smooth_calc']['smooth_window_len'] = window
		metadata['signal_smooth_calc']['polyorder'] = order
		
		save_exp_matrix(exp_matrix, obj)
		save_metadata(metadata, obj)

	
class gen_stim_by_traj(object):
	"""
	Class for generating the stimulus experienced by fly in its antenna region
	"""
	
	def __init__(self, subdir='2019-8-3', ant_size_mm=0.5, ant_dist_mm=1.5, 
				 sub_sample=4):
		"""
		Set the list of directories and antenna properties. Must first 
		transform image from pixels to [-1, 1], via:

		x' = 2x/proj_x_res - 1  
		y' = 1 - 2y/proj_y_res 
		
		or:
		
		X' = SX + T, 
		
		  where 
		
		S = [[2/proj_x_res, 0], [0, -2/proj_y_res]]
		T = [-1, 1].

		The y-direction is flipped by the image saving (i.e. OpenCV flips the 
		frames). Combining this affine transformation with the transformation
		from normalized stimulus units to camera pixels (given by A11,... B1, 
		etc. in the metadata)

		C = A*(SX + T) + B
		  = A'X + B', 
		  
		  where C is camera pixels, S is stimulus pixels, and 
		  using that S is diagonal:

		A' = [[A11*S11, A12*S22], [A21*S11, A22*S22]]
		B' = [A11*T1 + A12*T2 + B1, A21*T1 + A22*T2 + B2]

		A' and B' are represented in an affine matrix by matrix M. This matrix
		converts stimulus units (-1 to 1) to camera pixels. This allows us
		to deterimne where the the projected stimulus lands in relation to where
		the fly is located.
		"""
		
		self.exp_dirs = glob("%s/%s/*/" % (get_data_dir(), subdir))
		self.ant_size_mm = ant_size_mm
		self.ant_dist_mm = ant_dist_mm
		self.sub_sample = sub_sample
		
	def gen_stim_all_exp_dirs(self):
		"""
		Generate stimulus in antennas of flies in all experimental directories
		"""
		
		for exp_dir in self.exp_dirs:
			self.gen_stim_single_exp_dir(exp_dir)
		
	def gen_stim_single_exp_dir(self, exp_dir):
		"""
		Generate stimulus in antennas of flies in single directory.
		"""
		
		from psychopy import visual

		obj, flies, shapes = load_exp_data(exp_dir)

		ant_size_px = int(self.ant_size_mm/obj.mm_per_px)
		ant_dist_px = int(self.ant_dist_mm/obj.mm_per_px)
		
		# Fly state matrix for flies active at least one frame
		states = np.asarray(flies['states'])
		num_active_flies = np.sum(np.prod(1 - states, axis=0) == 0)
		states_active = states[:, :num_active_flies]
		
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

		exp_matrix = load_exp_matrix(obj)
		metadata = load_metadata(obj)
		traj_ranges = get_traj_ranges(exp_matrix)
		x = exp_matrix['x_smooth']/obj.mm_per_px
		y = exp_matrix['y_smooth']/obj.mm_per_px
		theta = exp_matrix['theta_smooth']*np.pi/180.

		# Antenna position; ensure does not go outside viewable region
		ant_x = x + np.cos(theta)*ant_dist_px
		ant_y = y + np.sin(theta)*ant_dist_px
		ant_x[ant_x > cam_x_res] = np.nan
		ant_y[ant_y > cam_y_res] = np.nan
		ant_x[ant_x < 0] = np.nan
		ant_y[ant_y < 0] = np.nan
		
		# This is the conversion matrix from proj to cam (see docstring)
		M = [[ 2*ainvxx/proj_x_res, 
			  -2*ainvxy/proj_y_res, 
			  -ainvxx + ainvxy + binvx], 
			 [ 2*ainvyx/proj_x_res, 
			  -2*ainvyy/proj_y_res, 
			  -ainvyx + ainvyy + binvy]] 
		M = np.array(M)
		
		if obj.stim.invert == False:
			bck_color = 'black'
		else:
			bck_color = 'red'
		
		win = visual.Window(size=[proj_x_res, proj_y_res],
							fullscr=False, screen=0, allowGUI=False, 
							allowStencil=False, monitor='testMonitor', 
							color=bck_color, blendMode='avg', useFBO=False,
							waitBlanking=False, gammaErrorPolicy='ignore')

		# Holds current index of exp_matrix for ith fly
		idxs = [0]*num_active_flies
		
		# Temporary stimulus files for each fly; will be deleted.
		data_fi = []
		for i, line in enumerate(range(num_active_flies)):
			filename = "%s/tmp_fly_data_%s.txt" % (exp_dir, i)
			try:
				os.remove(filename)
			except FileNotFoundError:
				pass
			data_fi.append(open(filename, 'ab'))
	
		# To hold saved movie of stimulus; save at lower frame rate for space
		filename = os.path.join(exp_dir, 'stim.mp4')
		stim_vid = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),
					obj.rec_rate//self.sub_sample, 
					(int(proj_x_res), int(proj_y_res)), 1)
		
		for frm_num in range((obj.rec_frms+obj.pre_rec_frms + 1)//obj.framepack_num):
			
			print(frm_num)
			
			# These hold all the arguments to regenerate ElementArrayStim
			kwargs = shapes[frm_num]
			
			# This messes with color for some reason -- bug in psychopy?
			del kwargs['rgbs']
			
			# Drawing shapes is the time bottleneck
			_shapes = visual.ElementArrayStim(win, **kwargs)
			_shapes.draw()
			win.flip()
			
			# Disk usage is higher but memory lower
			win.getMovieFrame(buffer='front')
			_, exp_dir_str = os.path.split(os.path.split(exp_dir)[0])
			win.saveMovieFrames('%s_tmp_frm_%s.bmp' % (exp_dir_str, frm_num))
			frame = cv2.imread(r'%s_tmp_frm_%s.bmp' % (exp_dir_str, frm_num))
			
			# Save stimulus video at reduced rate
			if frm_num % self.sub_sample == 0:
				stim_vid.write(frame)
			
			# opencv uses BGR; flip BGR --> RGB
			frame = frame[:, :, ::-1]
			
			# This is the conversion from normalized stimulus to cam units.
			shapes_conv = cv2.warpAffine(frame, M, (cam_x_res, cam_y_res))
			
			# Shapes box-averaged to give antenna size 
			blur_shapes = cv2.blur(shapes_conv, (ant_size_px, ant_size_px))
			
			for fly_num in range(num_active_flies):
				if states_active[frm_num, fly_num] == 1:
					exp_matrix_idx = traj_ranges[fly_num][0] + idxs[fly_num]
					_x = ant_x[exp_matrix_idx]
					_y = ant_y[exp_matrix_idx]
					if np.isfinite(_x)*np.isfinite(_y):
						np.savetxt(data_fi[fly_num], [blur_shapes[int(_y), 
						  int(_x)]], fmt='%d')
					else:
						np.savetxt(data_fi[fly_num],
						  [[np.nan, np.nan, np.nan]])
					idxs[fly_num] += 1

			try:
				os.remove('%s_tmp_frm_%s.bmp' % (exp_dir_str, frm_num))
			except FileNotFoundError:
				pass
			
		stim_vid.release()
		win.close()
		
		# Close saved stim files for writing
		for i in range(num_active_flies):
			data_fi[i].close()
		
		# Reopen files for reading, import all data to stim array
		stim = None
		for i in range(num_active_flies):
			filename = "%s/tmp_fly_data_%s.txt" % (exp_dir, i)
			data_file = open(filename, 'rb')
			data = np.loadtxt(data_file)
			if stim is None:
				if data.shape[0] != 0:
					stim = data
			else:
				if data.shape[0] != 0:
					stim = np.vstack((stim, data))
			data_file.close()
			os.remove(filename)
		
		exp_matrix['signal'] = stim
		metadata['stim_calc'] = dict()
		metadata['stim_calc']['ant_size_mm'] = self.ant_size_mm
		metadata['stim_calc']['ant_dist_mm'] = self.ant_dist_mm
		
		save_exp_matrix(exp_matrix, obj)
		save_metadata(metadata, obj)


class get_fly_properties():
	"""
	Calculate various properties of flies, such as whether they are behaving,
	grooming, etc.
	"""
	
	def __init__(self, subdir='2019-8-3',):
		"""
		Just get the experimental directories
		"""
		
		self.exp_dirs = glob("%s/%s/*/" % (get_data_dir(), subdir))
		
	def get_behaving_all_exp_dirs(self, min_dist=20, min_displ=5):
		"""
		Get all behaving flies for all dirs in a subdir
		"""
		
		for exp_dir in self.exp_dirs:
			self.get_behaving_single_exp_dir(exp_dir, min_dist, min_displ)
			
	def get_behaving_single_exp_dir(self, exp_dir, min_dist, min_displ):
		"""
		Get all behaving flies for single trial.
		"""
		
		print(exp_dir)
		obj, _, _ = load_exp_data(exp_dir)
		exp_matrix = load_exp_matrix(obj)
		metadata = load_metadata(obj)
		
		traj_ranges = get_traj_ranges(exp_matrix)
		x = exp_matrix['x_smooth']
		y = exp_matrix['y_smooth']
		behaving = np.zeros(len(x), dtype='bool')
	
		for rng in traj_ranges:
			if len(rng) == 0:
				continue
			dx = np.diff(x[rng])
			dy = np.diff(y[rng])
			traversed_dist = np.nansum(np.sqrt(dx**2.0 + dy**2.0))
			max_x_displ = abs(max(x[rng]) - min(x[rng]))
			max_y_displ = abs(max(y[rng]) - min(y[rng]))
			#if np.prod(np.isfinite(x[rng]*y[rng])) == 0:
			#	traversed_dist = 0
			
			# Considered behaving if total distance traveled is enough 
			# and if displacement in x- or y- direction is enough
			if (traversed_dist > min_dist) and ((max_x_displ > min_displ)
			  or (max_y_displ > min_displ)):
				behaving[rng] = True
			else:
				behaving[rng] = False
			
		exp_matrix['behaving'] = behaving
		metadata['behaving_min_dist'] = min_dist
		metadata['behaving_min_displ'] = min_displ
		
		save_exp_matrix(exp_matrix, obj)
		save_metadata(metadata, obj)

		return exp_matrix
