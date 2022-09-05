"""
Functions for manipulating the projected stimuli, including calibrating 
the projector with the camera.

Created by Nirag Kadakia at 18:03 6-25-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from psychopy import visual
from cam_funcs import init_cam, detect_cam_fps, grab_imgs
from track_funcs import detect_fly
from load_save_data import load_settings, save_settings
		

def set_proj_window():
	"""
	Define psychopy window for projection region
	
	Returns
	-------
	
	win: Psychopy window object
		Screen window on which to project
	"""
	
	config = load_settings()
	
	proj_x_res = config.getfloat("Projector", "res_dx")
	proj_y_res = config.getfloat("Projector", "res_dy")
	screen_num = config.getint("Projector", "screen")
	
	win = visual.Window(size=[proj_x_res, proj_y_res], fullscr=True, 
						screen=screen_num, allowGUI=False, 
						allowStencil=False, monitor='testMonitor', 
						color='black', blendMode='avg', useFBO=True)
		

	return win

class calibrate_proj():
	"""
	Class for calibrating projector with camera. Projects and then detects
	random points in the viewable region, then fits to an affine transform.
	
	Attrs
	-------
	
	config: configparser.ConfigParser object
		Loaded configuration of the experiment.
	cam_x0, cam_y0, cam_dx, cam_dy: floats
		offets and width/height of camera viewable region.
	c: PyCapture2.camera() object
		Camera object
	win: Psychopy window
		Projector object for stimuli.
	detector: detect_fly() object
		Object for detecting flies.
	stim_xs, stim_ys: lists
		Hold all locations of projected objects in stimulus units.
	detect_xs, detect_ys: lists
		Hold all locations of detected objected in camera units.
	iterations: int
		Number of points to use for fit.
	num_attempt: int
		Current projected stimuli (out of iterations).
	"""
	
	def __init__(self, iterations=100, blob_threshold=20, blob_min_area=60,
				 blob_max_area=800):
		"""
		Initialize projector, camera, and vals.
		
		Args
		-------
		
		iterations: int
			Number of points to use for fit.
		blob_threshold: float
			Threshold to use for detection -- should be weak.
		blob_min_area, blob_min_area: floats
			Min and max of blob area in pixels^2. MAx area set to avoid central 
			blob from projector itself.
		"""
		
		self.config = load_settings()
		self.cam_x0 = self.config.getfloat("Camera", "x0")
		self.cam_y0 = self.config.getfloat("Camera", "y0")
		self.cam_dx = self.config.getfloat("Camera", "dx")
		self.cam_dy = self.config.getfloat("Camera", "dy")
		
		# Load up camera
		self.c = init_cam(self.cam_x0, self.cam_y0, self.cam_dx, self.cam_dy)
		rate = detect_cam_fps(self.c)
		self.c.startCapture()
		
		# Load projector window through psychopy
		self.win = set_proj_window()
		
		# Set up the blob detector for weak stimuli. This will try to 
		# avoid detecting the light from the projector, by upper bounding
		# the blob area.
		self.detector = detect_fly(min_threshold=blob_threshold, 
								   min_area=blob_min_area, 
								   max_area=blob_max_area, 
								   min_circularity=0.2)
		
		# To hold the x and y positions, from -1 to 1, of the stimuli
		self.stim_xs = []
		self.stim_ys = []
		
		# To hold the x and y positions, in camera pixels, of the detections
		self.detect_xs = []
		self.detect_ys = []
		
		# Number of successful iterations to do
		self.iterations = iterations
		self.num_attempt = 0
		
	def project_and_detect(self):
		"""
		Update the projector with a blob and detect it using detection code.
		"""
		
		# Small circle anywhere in the arena
		proj_x_pct = np.random.uniform(-1, 1)
		proj_y_pct = np.random.uniform(-1, 1)
		blob = visual.Circle(self.win, radius=0.01)
		blob.pos = [proj_x_pct, proj_y_pct]
		blob.fillColor = 'white'
		blob.lineColor = 'white'
		blob.draw()
		
		# Draw blobs; wait momentarily to allow image to appear
		self.win.flip()
		time.sleep(0.1)
		
		# Grab the image and invert
		im = grab_imgs(self.c, 1)
		im_inv = 255 - im
		
		# Detect blob location only if 1 blob is detected!
		keypoints = self.detector.detect(im_inv)
		if len(keypoints) != 1:
			return
				
		self.stim_xs.append(proj_x_pct)
		self.stim_ys.append(proj_y_pct)
		self.detect_xs.append(keypoints[0].pt[0])
		self.detect_ys.append(keypoints[0].pt[1])
		
		self.num_attempt += 1
		print(self.num_attempt)
		
	def f_cam_to_proj(self, X):
		"""
		Cost function assuming detector-->stim related by affine transform.
		
		Args
		-------
		
		X: 6-element list
			Coefficients of affine transformation.
		"""
		
		Axx = X[0]
		Axy = X[1]
		Ayx = X[2]
		Ayy = X[3]
		Bx = X[4]
		By = X[5]
		
		cost = 0
		for iT in range(len(self.detect_xs)):
			cost += (Axx*self.detect_xs[iT] + Axy*self.detect_ys[iT] + 
					 Bx - self.stim_xs[iT])**2
			cost += (Ayx*self.detect_xs[iT] + Ayy*self.detect_ys[iT] + 
					 By - self.stim_ys[iT])**2
			
		return cost
	
	def f_proj_to_cam(self, X):
		"""
		Cost function assuming stim-->detector related by affine transform.
		
		Args
		-------
		
		X: 6-element list
			Coefficients of affine transformation.
		"""
		
		Ainvxx = X[0]
		Ainvxy = X[1]
		Ainvyx = X[2]
		Ainvyy = X[3]
		Binvx = X[4]
		Binvy = X[5]
		
		cost = 0
		for iT in range(len(self.stim_xs)):
			cost += (Ainvxx*self.stim_xs[iT] + Ainvxy*self.stim_ys[iT] + 
					 Binvx - self.detect_xs[iT])**2
			cost += (Ainvyx*self.stim_xs[iT] + Ainvyy*self.stim_ys[iT] + 
					 Binvy - self.detect_ys[iT])**2
			
		return cost
	
	def run(self):
		"""
		Run calibration by randomly projecting and detecting points.
		"""
		
		self.num_attempt = 0
		while self.num_attempt < self.iterations:
			self.project_and_detect()
		self.detect_xs = np.array(self.detect_xs)
		self.detect_ys = np.array(self.detect_ys)
		
		# Fit transform from camera to projector
		res = minimize(self.f_cam_to_proj, x0=np.zeros(6), method='L-BFGS-B')
		
		# Save estimated transformation matrix elements to file: cam -> proj
		self.config.set("Projector", "calibration_axx", '%.5e' % res.x[0])
		self.config.set("Projector", "calibration_axy", '%.5e' % res.x[1])
		self.config.set("Projector", "calibration_ayx", '%.5e' % res.x[2])
		self.config.set("Projector", "calibration_ayy", '%.5e' % res.x[3])
		self.config.set("Projector", "calibration_bx", '%.5e' % res.x[4])
		self.config.set("Projector", "calibration_by", '%.5e' % res.x[5])
		
		# This angle represents the rotation needed to align the x- and y-axes
		# between the camera and projector. Since stimulus units in psychopy
		# are a right-handed coordinate system (x is right, y is up), 
		# while the camera is a left-handed coordinate system (x is right, y
		# is down), then to transform between cam to proj, one first flips 
		# over the y-axis, then rotates by theta_stim_cam, then stretches, then
		# translates. Thus the angle transformation is 
		# theta --> 180 - theta + theta_stim_cam. This works for both
		# cam to proj and proj to cam. dtheta/dt --> -dtheta/dt
		# If the transformation has shear (up to 2 deg tolerance), then 
		# an angle cannot be defined and this is skipped
		angle_xy_xx = np.arctan(-res.x[1]/res.x[0])*180/np.pi
		angle_yx_yy = np.arctan(res.x[2]/res.x[3])*180/np.pi
		
		# Plot to check
		fit_x = res.x[0]*self.detect_xs + res.x[1]*self.detect_ys + res.x[4]
		fit_y = res.x[2]*self.detect_xs + res.x[3]*self.detect_ys + res.x[5]
		plt.scatter(self.detect_xs, self.stim_xs)
		plt.scatter(self.detect_ys, self.stim_ys)
		plt.plot(self.detect_xs, fit_x)
		plt.plot(self.detect_ys, fit_y)
		plt.show()
		
		# Fit the inverse and save
		res = minimize(self.f_proj_to_cam, x0=np.zeros(6), method='L-BFGS-B')
		self.config.set("Projector", "calibration_ainvxx", '%.5e' % res.x[0])
		self.config.set("Projector", "calibration_ainvxy", '%.5e' % res.x[1])
		self.config.set("Projector", "calibration_ainvyx", '%.5e' % res.x[2])
		self.config.set("Projector", "calibration_ainvyy", '%.5e' % res.x[3])
		self.config.set("Projector", "calibration_binvx", '%.5e' % res.x[4])
		self.config.set("Projector", "calibration_binvy", '%.5e' % res.x[5])
		if abs(angle_xy_xx - angle_yx_yy) > 2:
			print("Camera to projector has shear, cannot save "
					"theta_stim_cam to config file")
			self.config.set("Projector", "calibration_theta_stim_cam", 'NaN')
		else:
			self.config.set("Projector", "calibration_theta_stim_cam", '%.5e' 
							% angle_xy_xx)
		
		save_settings(self.config)
		
		self.c.stopCapture()
