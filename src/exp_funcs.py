"""
Multithreading tracking for recording and projector

Created by Nirag Kadakia at 13:52 01-04-2018
This work is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License.
To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
sys.path.append(os.path.join('..', 'src'))
from track_funcs import detect_fly, CentroidTracker, track_and_identify
from load_save_data import load_settings, save_exp_data, set_exp_dir, \
						   agg_shape_data
from stimuli import *
from odor_wind_protocols import *
from utils import get_git_commit, copy_file


class rec_and_proj():
	"""
	Class for recording data and delivering stimuli in real-time

	Attrs
	-------

	TODO

	"""

	def __init__(self, fly_area=5, detector_res_dec_fact=2,
				 rec_rate_fps=60, rec_length=120, pre_rec_length=0,
				 max_flies=1000, max_disappeared_frms=10, bypass_cam=False,
				 red_img_res=False, init_T=3.0,
				 stimulus_protocol=box_on_fly,
				 stimulus_protocol_kwargs=(), odor_wind_protocol=None,
				 odor_wind_protocol_kwargs=(), framepack=False,
				 exp_name='test', disp_frm_skips_lens=True,
				 run_exp_filepath=None):
		"""
		Initialize projector and camera.

		Args
		-------

		fly_area: float
			Minimum area of fly in mm^2
		detector_res_dec_fact: int
			Factor by which to reduce the resolution of the recorded
			image for object tracking. This will speed up object recognition.
		rec_rate_fps: float
			Recording rate. This must match the refresh rate of the
			projector, or there will be timing issues.
		rec_length: float
			Recording length of the experimental stimuli video in seconds.
		pre_rec_length: float
			Recording length before initiating stimuli recording in seconds.
		max_flies: int
			Maximum number of unique flies that can be ID'd in one video.
		max_disappeared_frms: int
			Maximum num frames a fly can be not recognized to be deregistered
		bypass_cam: bool
			If True, bypass camera and just use a dummy img (e.g. for testing)
		init_T: float
			Initialization time to wait before recording data
		stimulus_protocol: object of class stimuli.stim_protocol
			stimulus protocol to deliver, may depend on fly locations
		stimulus_protocol_kwargs: 1-element tuple
			Keyword Args required by the stimulus protocol object. Within
			the tuple is a dictionary of the keywords and their values
		framepack: bool
			If true, will deliver monocrhome stimuli 3X as fast, packed into
			separate RGB channels.
		exp_name: str
			Base name of experiment. If already used, a number will be
			appended to the end. All data are saved in dated folders.
		disp_frm_skips_lens: bool
			Plot the frame lengths and rec-proj latencies on a plot at the
			end of the experiment. If False, will just display in terminal.
		"""

		# Get projector and camera parameters
		self.config = load_settings()

		self.proj_x_res = self.config.getfloat("Projector", "res_dx")
		self.proj_y_res = self.config.getfloat("Projector", "res_dy")
		self.screen_num = self.config.getint("Projector", "screen")
		self.cam_x0 = self.config.getfloat("Camera", "x0")
		self.cam_y0 = self.config.getfloat("Camera", "y0")
		self.cam_dx = self.config.getfloat("Camera", "dx")
		self.cam_dy = self.config.getfloat("Camera", "dy")
		self.mm_per_px = self.config.getfloat("Camera", "mm_per_px")

		# Load up camera
		self.bypass_cam = bypass_cam
		if self.bypass_cam == False:
			from cam_funcs import init_cam, detect_cam_fps, grab_imgs
			self.c = init_cam(self.cam_x0, self.cam_y0, self.cam_dx, self.cam_dy)
			self.cam_rate = detect_cam_fps(self.c)
			self.c.startCapture()

		# Factor at which to reduce the pixel resolution for blob detection
		self.det_red_fact = detector_res_dec_fact

		# Detector and tracking functions depends on fly size
		self.max_flies = max_flies
		self.detector = detect_fly(min_area=fly_area/self.mm_per_px**2.0/
								   self.det_red_fact**2)
		self.ct = CentroidTracker(max_flies=self.max_flies,
								  max_disappeared_frms=max_disappeared_frms)
		self.fly_pos_vec = []
		self.fly_states_vec = []

		# Stimulus_protocol is a child class of stim_protocol, in stimuli.py
		self.stim_protocol = stimulus_protocol
		self.stim_kwargs = stimulus_protocol_kwargs
		self.shapes_vec = []

		# If set, child class of odor_wind_protocol in odor_wind_protocols.py
		if odor_wind_protocol is not None:
			self.odor_wind_protocol = odor_wind_protocol
			self.odor_wind_kwargs = odor_wind_protocol_kwargs

		self.framepack = framepack
		self.framepack_num = int(framepack*2 + 1)

		self.rec_rate = rec_rate_fps
		self.rec_frms = int(rec_length*self.rec_rate*self.framepack_num)
		self.rec_img = None

		self.pre_rec_length = pre_rec_length
		self.pre_rec_frms = int(pre_rec_length*self.rec_rate*self.framepack_num)

		self.rec_times = []
		self.proj_times = []
		self.frame_dts = None
		self.init_T = init_T
		self.init_frms = int(self.rec_rate*self.init_T*self.framepack_num)

		# Experiment name for saving
		self.exp_name = exp_name
		self.out_dir, self.exp_dir = set_exp_dir(self.exp_name)

		# copy file used to run experiment
		self.run_exp_filepath = run_exp_filepath
		if self.run_exp_filepath is not None:
			self.exp_filepath = copy_file(src=self.run_exp_filepath, dest=self.out_dir)

		# git information for future reproducibility
		self.branch, self.commit = get_git_commit()

		# Plot frame lengths at end? (to ensure minimal skipped frames)
		self.disp_frm_skips_lens = disp_frm_skips_lens

		# Video container
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out_file = os.path.join(self.out_dir, 'frames.avi')
		self.video  = cv2.VideoWriter(out_file, fourcc, self.rec_rate,
						(int(self.cam_dx), int(self.cam_dy)), False)

	def thread_proj(self):
		"""
		The win.flip() keeps the timing precise at the frame rate of 60 Hz.
		The end of the flip command calls one loop of the recording thread,
		which then runs concurrently with the next stimulus update and next flip.
		"""

		# Need to import pyglet from this thread -- cannot cross threads
		from psychopy import visual
		from psychopy.visual.windowframepack import ProjectorFramePacker
		win = visual.Window(size=[self.proj_x_res, self.proj_y_res],
							fullscr=False, screen=self.screen_num,
							allowGUI=False, allowStencil=False,
							monitor='testMonitor', color='black',
							blendMode='avg', useFBO=False,
							waitBlanking=True)

		# Pack stimuli into 3 separate channels
		if self.framepack == True:
			ProjectorFramePacker(win)

		# Initialize the the odor/wind protocol if set:
		has_odor_wind_protocol = hasattr(self, 'odor_wind_protocol')
		if has_odor_wind_protocol:
			self.odor_wind = self.odor_wind_protocol(self.rec_rate,
													 **self.odor_wind_kwargs)

		# Initialize the stimulus protocol, initial shapes
		self.stim = self.stim_protocol(win, self.rec_rate, self.max_flies,
									   framepack=self.framepack,
									   **self.stim_kwargs)
		self.stim.initialize_shapes()

		# Initial image if camera being bypassed
		if self.bypass_cam == True:
			self.cap = cv2.VideoCapture(os.path.join('..', 'data', 'test_frames.avi'))
			self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			_, self.rec_img = self.cap.read()

		# Initialization; this will run for init_T seconds to warm up;
		# no data is being saved
		print(f'Initializating for {self.init_T} seconds...')
		self.thread_rec(frm_num=-1)
		for frm_num in np.arange(-self.init_frms, 0):
			win.clearBuffer()
			if frm_num % self.framepack_num == 0:
				thread_rec = Thread(target=self.thread_rec, args=(-1, ))
			self.stim.update(self.ct, self.centroids, self.states, frm_num)
			win.flip()
			if frm_num % self.framepack_num == 0:
				thread_rec.start()

		# Initial flip; set beginning time; record initial objects
		win.recordFrameIntervals = True
		if self.framepack == True:
			win.refreshThreshold  = 0.020
		else:
			win.refreshThreshold  = 0.021

		# Clear stimulus after warm up
		self.stim.shapes.setAutoDraw(False)
		win.flip()

		self.time_beg = time.perf_counter()
		self.thread_rec(frm_num=0)
		self.shapes_vec.extend((agg_shape_data(self.stim.shapes),))

		print(f'Pre-recording for {self.pre_rec_length} seconds...')
		# Begin looping proj --> record/track
		for frm_num in range(1, self.pre_rec_frms + self.rec_frms + 1):
			win.clearBuffer()

			# Update the odor/wind protocol if set
			if has_odor_wind_protocol:
				self.odor_wind.update(frm_num)

			# Compute framepack condition only once
			frm_num_mod0_framepack_num = frm_num % self.framepack_num == 0

			# Update the projected stimuli shapes
			if frm_num_mod0_framepack_num:
				thread_rec = Thread(target=self.thread_rec, args=(frm_num, ))

			# Update stimuli only after pre-recording time is over
			if frm_num > (self.pre_rec_frms + 1):
				# offset pre-recording frames
				frm_num = frm_num - self.pre_rec_frms
				self.stim.update(self.ct, self.centroids, self.states, frm_num)
			elif frm_num == (self.pre_rec_frms + 1):
				# Pre-recording time is up, restore auto draw for stimuli shapes
				print(f'Running exp for {self.rec_frms//self.rec_rate} '
					  f'seconds...')
				self.stim.shapes.setAutoDraw(True)

			if frm_num_mod0_framepack_num:
				self.shapes_vec.extend((agg_shape_data(self.stim.shapes),))

			win.flip()

			# Spawn recording thread at given recording rate
			if frm_num_mod0_framepack_num:
				if frm_num < (self.pre_rec_frms + self.rec_frms + 1):
					thread_rec.start()

				self.proj_times.append(time.perf_counter() - self.time_beg)
			
				# print('%.3f' % (time.perf_counter() - self.time_beg))

				# check if nothing was detected
				# prevent running experiments if lighting is not ideal
				if self.centroids.any() == False:
					print("no flies detected")
		
		print('%s dropped frames....' % win.nDroppedFrames)

		self.frame_dts = win.frameIntervals
		win.close()

		if has_odor_wind_protocol:
			self.odor_wind.terminate()

	def thread_rec(self, frm_num):
		"""
		Frame recording, object ID'ing and tracking, and frame saving thread.

		Args
		-------

		frm_num: int
			If -1, this is the initialization phase and nothing will be saved
		"""

		if self.bypass_cam == False:
			from cam_funcs import grab_imgs
			self.rec_img = grab_imgs(self.c, 1)

		if frm_num > -1:
			if self.bypass_cam == True:
				_, self.rec_img = self.cap.read()

			## TODO: the output is unreadable for images loaded from disk.??
			self.video.write(self.rec_img)

		# Update fly positions based on recorded image
		self.centroids, self.states = track_and_identify(self.rec_img,
										self.det_red_fact,
										self.detector, self.ct)
		if (frm_num > -1):
			self.fly_pos_vec.extend((self.centroids.copy(),))
			self.fly_states_vec.extend((self.states.copy(),))
			self.rec_times.extend((time.perf_counter() - self.time_beg,))

	def run(self):
		"""
		Run experiment in projector thread, which calls recording
		thread at each projector flip.
		"""

		thread_proj = Thread(target=self.thread_proj)
		thread_proj.start()
		thread_proj.join()
		self.video.release()

		print('Waiting a few seconds for recording threads to terminate...')
		time.sleep(3.0)

		# Plot latency b/w each recorded time and each projected frame
		# and each recorded and projected frame time.
		if self.disp_frm_skips_lens:
			self.rec_times = np.array(self.rec_times)
			self.proj_times = np.array(self.proj_times)
			fig = plt.figure()
			plt.subplot(211)
			plt.plot(self.rec_times, label='recorded', linestyle='--')
			plt.plot(self.proj_times, label='projected', linestyle=':')
			plt.legend()
			if len(self.rec_times) == len(self.proj_times):
				plt.subplot(212)
				plt.plot(self.rec_times - self.proj_times, label='recorded - projected)')
				plt.show()
			else:
				print(f'{len(self.rec_times)} recorded frames but '
					  f'{len(self.proj_times)} projected frames!')
			plt.plot(self.frame_dts)
			plt.show()

		save_exp_data(self)
