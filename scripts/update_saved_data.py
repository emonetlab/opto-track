"""
Script to update any saved object files with new attributes 
that were added in newer versions of the code. Typically this will
apply to stimulus attributes that were not saved previously.

First data sets saved the stim_kwargs rather than the actual 
stim attributes. This ammends that so there is no ambiguity in 
what stimulus was presented. Especially since some args have now
been changed to keyword arguments.

Created by Nirag Kadakia at 13:55 10-10-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import sys, os
import configparser
import ntpath
import gzip
from psychopy import visual
sys.path.append('../src')
from gen_fly_data_from_vid import *
from stimuli import *
from load_save_data import *


def update_config(subdirs):
	"""
	Update obj.config with any new experimental attributes
	"""
	
	for subdir in subdirs:
			   
		a = gen_xy_theta(subdir)
		for dir in a.exp_dirs:
			print(subdir, dir[-30:])
			obj, _, _ = load_exp_data(dir)

			# Here, we add projector spatial resolution in x and y which 
			# was not saved previously..
			obj.config.set('Projector', 'mm_per_px_x', '%s' % .2057)
			obj.config.set('Projector', 'mm_per_px_y', '%s' % .2057)
			
			# Here we add theta, which is not 
			axx = obj.config.getfloat('Projector', 'calibration_axx')
			axy = obj.config.getfloat('Projector', 'calibration_axy')
			ayx = obj.config.getfloat('Projector', 'calibration_ayx')
			ayy = obj.config.getfloat('Projector', 'calibration_ayy')
			
			theta_1 = np.arctan(-axy/axx)*180/np.pi
			theta_2 = np.arctan(ayx/ayy)*180/np.pi
			
			if abs(theta_1 - theta_2) < 2:
				obj.config.set("Projector", "calibration_theta_stim_cam", 
								'%.5e' % theta_1)
			else:
				obj.config.set("Projector", "calibration_theta_stim_cam", 'NaN')
			print(obj.config.getfloat('Projector', 'calibration_theta_stim_cam'))
			
			out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
			out_file = '%s/obj.pklz' % out_dir
			with gzip.GzipFile(out_file, 'wb') as f:
				pickle.dump(obj, f)
			
			
def update_stim_obj(subdirs):
	"""
	Update stimuli objects with updated attributes by recreating stimulus 
	objects with correct kwargs, then resaving obj.
	"""
	
	for subdir in subdirs:
			   
		a = gen_xy_theta(subdir)
		for dir in a.exp_dirs:
			print(subdir, dir[-30:])
			obj, _, _ = load_exp_data(dir)
			
			win = visual.Window(size=[10, 10], 
								fullscr=False, screen=1, 
								allowGUI=False, allowStencil=False, 
								monitor='testMonitor', color='black', 
								blendMode='avg', useFBO=False, 
								waitBlanking=True) 
			win.close()
			
			# All data through 11-10 was not framepacked; used invert from kwargs
			obj.stim = obj.stim_protocol(win=win, rec_rate=60, max_flies=1000, 
									 framepack=False, **obj.stim_kwargs)
			del (obj.stim.win)
			
			out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
			out_file = '%s/obj.pklz' % out_dir
			with gzip.GzipFile(out_file, 'wb') as f:
				pickle.dump(obj, f)
			
def undo_framepack():
	"""
	If accidentally saved as framepack but not flipped at 3X
	"""
	
	subdir = '2019-11-18'
	dir_nums = [12, 13, 14, 15]
	a = get_fly_properties(subdir)
		
	for dir_num in dir_nums:
		obj, flies, shapes = load_exp_data(a.exp_dirs[dir_num])
		print(obj.exp_dir)
		
		obj.rec_frms = obj.rec_frms//3
		obj.init_frms = obj.init_frms//3
		shapes = shapes[::3]
		obj.frame_dts = obj.frame_dts[::3]
		obj.framepack = False
		obj.framepack_num = 1
		obj.stim.proj_speed = obj.stim.proj_speed*3
		obj.stim.framepack_num = 1
			
		
		out_dir = '%s/%s' % (get_data_dir(), obj.exp_dir)
		out_file = '%s/projected_shapes.pklz' % out_dir
		with gzip.GzipFile(out_file, 'wb') as f:
			pickle.dump(shapes, f)
		
		out_file = '%s/obj.pklz' % out_dir
		with gzip.GzipFile(out_file, 'wb') as f:
			pickle.dump(obj, f)
	
def fix_vert_bar_direction(subdirs):
	"""
	Old version of code accidentally had the wrong direction for
	the motion of vertical bars -- + referred to leftward.
	"""
	
	for subdir in subdirs:
			   
		a = gen_xy_theta(subdir)
		for dir in a.exp_dirs:
			obj, _, _ = load_exp_data(dir)
			if obj.stim_protocol == vert_moving_bars:
				pass
			else:
				continue
			print(obj.stim.width_stim_units, obj.stim.speed,
				   obj.stim.proj_speed)
			
			obj.stim.width_stim_units = abs(obj.stim.width_stim_units)
			obj.stim.speed = abs(obj.stim.speed)*np.sign(obj.stim.proj_speed)
			
			out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
			out_file = '%s/obj.pklz' % out_dir
			with gzip.GzipFile(out_file, 'wb') as f:
				pickle.dump(obj, f)
			
def fix_calibration(subdirs):
	""" 
	Forgot to calibrate? Update values manually.
	"""
	
	for subdir in subdirs:
	
		a = gen_xy_theta(subdir)
		for dir in a.exp_dirs:
			obj, _, _ = load_exp_data(dir)
			obj.config.set("Projector", "calibration_axx", 
							'%.5e' % -1.16175e-03)
			obj.config.set("Projector", "calibration_axy", 
							'%.5e' % -1.10587e-06)
			obj.config.set("Projector", "calibration_ayx", 
							'%.5e' % 2.48974e-07)
			obj.config.set("Projector", "calibration_ayy", 
							'%.5e' % 1.86456e-03)
			obj.config.set("Projector", "calibration_bx", 
							'%.5e' % 9.66067e-01)
			obj.config.set("Projector", "calibration_by", 
							'%.5e' % -1.02851e+00)
			
			obj.config.set("Projector", "calibration_ainvxx", 
							'%.5e' % -8.60775e+02)
			obj.config.set("Projector", "calibration_ainvxy", 
							'%.5e' % -5.00013e-01)
			obj.config.set("Projector", "calibration_ainvyx", 
							'%.5e' % 1.24366e-01)
			obj.config.set("Projector", "calibration_ainvyy", 
							'%.5e' % 5.36301e+02)
			obj.config.set("Projector", "calibration_binvx", 
							'%.5e' % 8.31036e+02)
			obj.config.set("Projector", "calibration_binvy", 
							'%.5e' % 5.51498e+02)
			
			obj.config.set("Projector", "calibration_theta_stim_cam ",  
							'%.5e' % -5.453e-02)
			
			out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
			out_file = '%s/obj.pklz' % out_dir
			with gzip.GzipFile(out_file, 'wb') as f:
				pickle.dump(obj, f)

def fix_name(subdir):
	"""
	Change the name of a recorded experiment. If you have changed
	the name of a directory manually, this goes in and changes the saved
	"exp_name" in the object file to match the directory name.
	"""
	
	a = gen_xy_theta(subdir)
	
	for dir in a.exp_dirs:
		obj, _, _ = load_exp_data(dir)
		
		# Remove trailing slash, get name from the changed folder
		head, tail = ntpath.split(dir)
		new_exp_dir = subdir + '/' + ntpath.basename(head)
		
		# Remove the _%d to get the experiment name without number
		new_exp_name = ntpath.basename(head).rsplit('_', 1)[0]
		
		# Overwrite if needed
		if (new_exp_name != obj.exp_name):
			obj.exp_name = new_exp_name
			print(obj.exp_name)
		if (new_exp_dir != obj.exp_dir):
			obj.exp_dir = new_exp_dir
			print(obj.exp_dir)
		
		out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
		out_file = '%s/obj.pklz' % out_dir
		with gzip.GzipFile(out_file, 'wb') as f:
			pickle.dump(obj, f)
		
def ammend_on_off_width_to_moving_bars(subdirs):
	"""
	Old moving bars script specified number of bars and width. New
	one specifies width of bar and width of blank.
	"""
	
	for subdir in subdirs:
		a = gen_xy_theta(subdir)
		for _dir in a.exp_dirs:
			obj, _, shapes = load_exp_data(_dir)
			print(subdir, _dir[-30:])
			if obj.stim_protocol == horiz_moving_bars:
				
				stim_to_mm = obj.stim.width/obj.stim.width_stim_units
				on_width = obj.stim.width
				dy = min(abs(np.diff(shapes[0]['xys'][:, 1])))
				cycle_width_stim_units = dy
				cycle_width = cycle_width_stim_units*stim_to_mm
				off_width = cycle_width - on_width
				obj.stim.on_width = on_width
				obj.stim.off_width = off_width
			
			elif obj.stim_protocol == vert_moving_bars:
				stim_to_mm = obj.stim.width/obj.stim.width_stim_units
				on_width = obj.stim.width
				dx = min(abs(np.diff(shapes[0]['xys'][:, 0])))
				cycle_width_stim_units = dx
				cycle_width = cycle_width_stim_units*stim_to_mm
				off_width = cycle_width - on_width
				
				obj.stim.on_width = on_width
				obj.stim.off_width = off_width
								
			else:
				continue
			
			out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
			out_file = '%s/obj.pklz' % out_dir
			with gzip.GzipFile(out_file, 'wb') as f:
				pickle.dump(obj, f)
			
def amend_old_full_field_flash_spec_class(subdirs):
	"""
	The old `full_field_flash_spec' is now deprecated. Any old
	stimuli using that classs can be updated to the attributes of
	the new class using this function
	"""
	
	for subdir in subdirs:
		a = gen_xy_theta(subdir)
		for _dir in a.exp_dirs:
			obj, _, shapes = load_exp_data(_dir)
			if obj.stim_protocol == full_field_flash_spec:
				print(subdir, _dir[-30:])
			
				if not 'dur' in dir(obj.stim):
				
					# These were done before 7/16/2020, in which case it only 
					# handled 50% duty cycle
					obj.stim.dur = 1./obj.stim.freq/2.
					obj.stim.duty_cycle = 0.5
				elif obj.stim.dur < 1:
				
					# If < 1, it represents duty cycle
					obj.stim.duty_cycle = obj.stim.dur
					obj.stim.dur = obj.stim.duty_cycle/obj.stim.freq
				else:
				
					# If > 1, then represens duration. Previous were saved 
					# in milliseconds, now in seconds
					obj.stim.duty_cycle = obj.stim.freq*obj.stim.dur/1000.
					obj.stim.dur = obj.stim.dur/1000.
					
				# All the new attributes
				obj.stim_protocol = full_field_flash
				obj.stim.num_flash_cycle_frms = int(obj.stim.rec_rate*
					obj.stim.framepack_num/obj.stim.freq)
				obj.stim.num_flash_frms = int(obj.stim.num_flash_cycle_frms*
					obj.stim.duty_cycle)
		
				obj.stim.On_T = obj.stim.stim_len
				obj.stim.Off_T = obj.stim.stim_len
				obj.stim.num_On_Off_frms = int((obj.stim.Off_T + obj.stim.On_T)
					*obj.stim.rec_rate*obj.stim.framepack_num)
				obj.stim.num_On_frms = int(obj.stim.On_T*obj.stim.rec_rate*
					obj.stim.framepack_num)
			
				print('freq = ', obj.stim.freq, 'dur = ', obj.stim.dur, 
					   'duty_cycle = ', obj.stim.duty_cycle)
			
			elif obj.stim_protocol == full_field_flash:
				
				# If it includes this key, it is the new function and don't
				# need to overwrite
				if 'num_On_frms' in dir(obj.stim):
					continue
					
				# Could only handle 50% duty cycle
				obj.stim.dur = 1./obj.stim.freq/2.
				obj.stim.duty_cycle = 0.5
			
				obj.stim_protocol = full_field_flash
				obj.stim.num_flash_cycle_frms = int(obj.stim.rec_rate*
					obj.stim.framepack_num/obj.stim.freq)
				obj.stim.num_flash_frms = int(obj.stim.num_flash_cycle_frms*
					obj.stim.duty_cycle)
		
				# Recording length (s) not saved, so get from frames and rate
				obj.stim.On_T = obj.rec_frms/obj.rec_rate/obj.framepack_num
				obj.stim.Off_T = obj.rec_frms/obj.rec_rate/obj.framepack_num
				obj.stim.num_On_Off_frms = int((obj.stim.Off_T + obj.stim.On_T)
					*obj.stim.rec_rate*obj.stim.framepack_num)
				obj.stim.num_On_frms = int(obj.stim.On_T*obj.stim.rec_rate*
					obj.stim.framepack_num)
			
				print('freq = ', obj.stim.freq, 'dur = ', obj.stim.dur, 
					   'duty_cycle = ', obj.stim.duty_cycle)
			
			else:
				print('skipping ...', subdir, _dir[-30:])
				continue
			
			out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
			out_file = '%s/obj.pklz' % out_dir
			with gzip.GzipFile(out_file, 'wb') as f:
				pickle.dump(obj, f)

def amend_scintillator_2_with_framepack(subdir):
	"""
	An old commit change (be3a723) updated framepacked stimuli at 1/3 the 
	rate that it should have. This persisted until the (bcfe1de) commit, 6
	months later. Fortunately, the majority of the data in this time was 
	not framepacked, so it did not affect many experimental datasets. 
	
	This function amends data taken in this time. This applies only to 
	scinilattor_2 datasets, which were the only framepacked datasets taken in 
	this time. 
	"""
	
	a = gen_xy_theta(subdir)
	for _dir in a.exp_dirs:
		obj, _, shapes = load_exp_data(_dir)
		if obj.stim_protocol == scintillator_2:
			print(subdir, _dir[-30:])
		
			# It has already been updated; skip
			if 'corr_x_step' in dir(obj.stim):
				continue
			if obj.framepack == True:
				obj.stim.corr_x_step = 3
				obj.stim.num_corrs = int(obj.stim.num_corrs/3)
				obj.stim.t_step *= 3
			else:
				obj.stim.corr_x_step = 1
			print(obj.stim.num_corrs, obj.stim.corr_x_step, obj.stim.t_step)
			
		out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
		out_file = '%s/obj.pklz' % out_dir
		with gzip.GzipFile(out_file, 'wb') as f:
			pickle.dump(obj, f)

def add_cycle_attrs_to_moving_bars(subdirs):
	"""
	Moving horiz and vert bars now have an `On_T' and `Off_T' attribute.
	Add this to old datasets which did not have this attribute in a previous
	version of the class.
	"""
	
	for subdir in subdirs:
			   
		a = gen_xy_theta(subdir)
		for dir in a.exp_dirs:
			obj, _, _ = load_exp_data(dir)
			if obj.stim_protocol == horiz_moving_bars:
				pass
			elif obj.stim_protocol == vert_moving_bars:
				pass
			else:
				continue
			if hasattr(obj.stim, 'On_T'):
				continue
			obj.stim.On_T = None
			obj.stim.Off_T = None
			
			out_dir = '%s/%s/' % (get_data_dir(), obj.exp_dir)
			out_file = '%s/obj.pklz' % out_dir
			with gzip.GzipFile(out_file, 'wb') as f:
				pickle.dump(obj, f)

subdirs = os.listdir(r'C:/Users/nk479/Dropbox '\
					'(emonetlab)/users/nirag_kadakia/data/'\
					'optogenetic-assay/assay/rec_videos')
#subdirs = ['2020-9-14']
#update_stim_obj(subdirs)
#update_config(subdirs)
#undo_framepack()
#fix_vert_bar_direction(subdirs)
#fix_calibration(subdirs)
#fix_name(subdir='2020-1-30')
#ammend_on_off_width_to_moving_bars(subdirs)
#subdirs = amend_old_full_field_flash_spec_class(subdirs)
#amend_scintillator_2_with_framepack('2020-3-12')
#add_cycle_attrs_to_moving_bars(subdirs)