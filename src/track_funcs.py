"""
Functions for identifying and tracking objs and 
their identities.

Created by Nirag Kadakia at 18:03 6-25-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
from scipy.signal import savgol_filter
from collections import OrderedDict
from scipy.spatial import distance as dist
import cv2
from utils import get_consecutive_lists, get_non_nan_splits


def detect_fly(min_threshold=80, min_area=60, 
			   max_area=1e7, min_circularity=0.5):
	"""
	Detect a fly in image using opencv obj recognition.
	
	Args
	-------
	
	min_threshold: float
		Minimum pixel intensity for obj detection.
	min_area: int
		Minimum area of obj in pixels^2 
	max_area:
		Maximum area of obj in pixels^2 
	min_circularity:
		Minimum circularity for fly obj
		
	Returns
	-------
	
	detector: obj instance
		openCV SimpleBlobDetector obj instance
	"""
	
	params = cv2.SimpleBlobDetector_Params()
	
	params.filterByCircularity = True
	params.filterByConvexity = False
	params.filterByArea = True
	params.minThreshold = min_threshold
	params.minArea = min_area
	params.maxArea = max_area
	params.minCircularity = min_circularity
	
	detector = cv2.SimpleBlobDetector_create(params)

	return detector

def track_and_identify(image, det_red_fact, detector, ct):
	"""
	Detect the objs from an image, and identify objs by their identity.
	This function uses a defined detector obj and tracker obj.
	
	Args
	-------
	
	image: (N, 2) array
		Array of pixel intensities of recorded image from camera.
	det_red_fact: int
		Image will be subsampled by factor det_red_fact in obj tracking.
	detector:
		detect_fly obj
	ct:
		centroidTracker obj instance
		
	Returns
	-------
	
	objs: list
		centroidTracker identified objs
	"""
	
	image_inv = 255 - image[::det_red_fact, ::det_red_fact]
	keypoints = detector.detect(image_inv)
	rects = []
	for iK in range(len(keypoints)):
		iP = keypoints[iK]
		rects.extend(([iP.pt[0]*det_red_fact - iP.size, 
					  iP.pt[1]*det_red_fact - iP.size,
					  iP.pt[0]*det_red_fact + iP.size, 
					  iP.pt[1]*det_red_fact + iP.size],))
	
	# Identify which is which from last frame
	centroids, state = ct.update(rects)
	
	return centroids, state


class CentroidTracker():
	"""
	Identify fly IDs by matching closest flies in subsequent frames.
	
	Attributes
	-------
	
	next_obj_ID: int
		Next ID for a newly identified obj
	objs: OrderedDict
		Key is fly ID and value is a numpy array of their position
	disappeared: OrderedDict
		Key is fly ID and value is number of frames for which it is
		disappeared.
	max_disappeared_frms: int
		Maximum number of frames a fly can be out of the frame before 
		a new ID is generated
	"""
	
	def __init__(self, max_flies, max_disappeared_frms=30):
		
		"""
		Identify fly IDs by matching closest flies in subsequent frames.
		
		Args
		-------
		
		max_flies: int
			Maximum number of flies that we can hold.
		max_disappeared_frms: int
			maximum number of frames a fly can be out of the frame before 
			a new ID is generated
			
		Returns
		-------
		
		objs: list
			centroidTracker identified objs
		"""
		
		self.next_obj_ID = 0
		self.objs = OrderedDict()
		self.disappeared = OrderedDict()
		self.max_disappeared_frms = max_disappeared_frms
		
		# Thee will actually hold the data to be returned. 
		self.centroid_array = np.zeros((max_flies, 2))
		self.state_array = np.zeros((max_flies))
		
	def register(self, centroid):
		"""
		Register new fly ID
		
		Args
		-------
		
		centroid: 2-element list
			Position of fly.
		"""
		
		self.objs[self.next_obj_ID] = centroid
		self.disappeared[self.next_obj_ID] = 0
		self.next_obj_ID += 1
	
	def deregister(self, obj_ID):
		"""
		Deregister fly if disappeared.
		
		Args
		-------
		
		obj_ID: int
			ID of fly that is to be de-registered.
		"""
		
		del self.objs[obj_ID]
		del self.disappeared[obj_ID]
		self.state_array[obj_ID] = 0
		
	def update(self, rects):
		"""
		Update the fly IDs and register/deregister flies
		
		Args
		-------
		
		rects: list of N 4-element lists
			N positions of flpy by lower x, upper x, lower y, upper y.
		"""
		
		# No detections -- everything has disappeared
		if len(rects) == 0:
			
			# Loop over any existing tracked objs and mark as disappeared
			for obj_ID in self.disappeared.keys():
				self.disappeared[obj_ID] += 1
				if self.disappeared[obj_ID] > self.max_disappeared_frms:
					self.deregister(obj_ID)
 
			return self.centroid_array, self.state_array
		
		input_centroids = np.zeros((len(rects), 2), dtype="int")
		for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
			
			cX = int((start_x + end_x) / 2.0)
			cY = int((start_y + end_y) / 2.0)
			input_centroids[i] = (cX, cY)
			
		if len(self.objs) == 0:
			for i in range(0, len(input_centroids)):
				self.register(input_centroids[i])
		else:
			
			# Grab the set of obj IDs and corresponding centroids
			obj_IDs = list(self.objs.keys())
			obj_centroids = list(self.objs.values())
 
			# Distance between each pair of current objs and detected objs
			D = dist.cdist(np.array(obj_centroids), input_centroids)
 
			# in order to perform this matching we must (1) find the smallest 
			# value in each row and then (2) sort the row indexes based on 
			# their minimum values so that the row with the smallest value 
			# is at the *front* of the index list
			rows = D.min(axis=1).argsort()
 
			# Next, we perform a similar process on the columns by finding 
			# the smallest value in each column and then sorting using 
			# the previously computed row index list
			cols = D.argmin(axis=1)[rows]		
			
			# In order to determine if we need to update, register,
			# or deregister an obj we need to keep track of which
			# of the rows and column indexes we have already examined
			used_rows = set()
			used_cols = set()
 
			# Loop over the combination of the (row, column) index tuples
			for (row, col) in zip(rows, cols):
				
				# If already examined either row or col value, ignore
				if row in used_rows or col in used_cols:
					continue
 
				obj_ID = obj_IDs[row]
				self.objs[obj_ID] = input_centroids[col]
				self.disappeared[obj_ID] = 0
				self.centroid_array[obj_ID] = input_centroids[col]
				self.state_array[obj_ID] = 1
				
				used_rows.add(row)
				used_cols.add(col)
			
			# Compute both the row and column index we have NOT yet examined
			unused_rows = set(range(0, D.shape[0])).difference(used_rows)
			unused_cols = set(range(0, D.shape[1])).difference(used_cols)
			
			# If number of objs is more than detected, see if some disappeared.
			if D.shape[0] >= D.shape[1]:
				
				# Loop over the unused row indexes
				for row in unused_rows:
					
					obj_ID = obj_IDs[row]
					self.disappeared[obj_ID] += 1
					if self.disappeared[obj_ID] > self.max_disappeared_frms:
						self.deregister(obj_ID)
						self.state_array[obj_ID] = 0
			
			# Otherwise, register each new detected obj as a new trackable obj
			else:
				for col in unused_cols:
					self.register(input_centroids[col])
		
		return self.centroid_array, self.state_array


class fit_ellipse():
	"""
	Least squares fitting of set of points to an ellipse
	"""
	
	def fit(self, data):
		
		x, y = np.asarray(data, dtype=float)

		#Quadratic part of design matrix [eqn. 15] from (*)
		D1 = np.mat(np.vstack([x**2, x*y, y**2])).T
		#Linear part of design matrix [eqn. 16] from (*)
		D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T
		
		#forming scatter matrix 
		S1 = D1.T*D1
		S2 = D1.T*D2
		S3 = D2.T*D2  
		
		#Constraint matrix 
		C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

		#Reduced scatter matrix 
		M = C1.I*(S1 - S2*S3.I*S2.T)

		#M*|a b c > = l|a b c >. Find eigs from this equation 
		eval, evec = np.linalg.eig(M) 

		# eigenvector must meet constraint 4ac - b^2 to be valid.
		cond = 4*np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
		a1 = evec[:, np.nonzero(cond.A > 0)[1]]
		
		#|d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
		a2 = -S3.I*S2.T*a1
		
		# eigenvectors |a b c d f g> 
		self.coef = np.vstack([a1, a2])
		self._save_parameters()
			
	def _save_parameters(self):
		
		#eigenvectors are the coefficients of an ellipse in general form
		#a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 
		a = self.coef[0,0]
		b = self.coef[1,0]/2.
		c = self.coef[2,0]
		d = self.coef[3,0]/2.
		f = self.coef[4,0]/2.
		g = self.coef[5,0]
		
		#finding center of ellipse [eqn.19 and 20] from (**)
		x0 = (c*d - b*f)/(b**2. - a*c)
		y0 = (a*f - b*d)/(b**2. - a*c)
		
		#Find the semi-axes lengths [eqn. 21 and 22] from (**)
		numerator = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f-a*c*g)
		denominator1 = (b*b - a*c)*((c - a)*np.sqrt(1. + 4.*b*b/\
						((a - c)*(a - c))) -(c + a))
		denominator2 = (b*b - a*c)*((a - c)*np.sqrt(1. + 4.*b*b/\
						((a - c)*(a - c))) - (c + a))
		width = np.sqrt(numerator/denominator1)
		height = np.sqrt(numerator/denominator2)
		phi = .5*np.arctan((2.*b)/(a - c))
		
		self._center = [x0, y0]
		
		if width > height:
			self._width = width
			self._height = height
			self._phi = phi*180/np.pi
		else:
			self._height = width
			self._width = height
			self._phi = phi*180/np.pi + 90

	@property
	def center(self):
		return self._center

	@property
	def width(self):
		return self._width

	@property
	def height(self):
		return self._height

	@property
	def phi(self):
		return self._phi

	def parameters(self):
		return self.center, self.width, self.height, self.phi		

def fix_ellipse_orientation(theta_raw, x, y, min_walk_spd, fps, 
							v_smooth_window, min_walk_dur=0.3):
	"""
	Correct the orientation vector from ellipse fitting, which may be 
	off by 180 degrees due to 2-fold symmetry. This method uses the 
	heading of the fly at walking times to figure out 

	Args
	-------

	theta_raw: length-N numpy array
		Recorded orientation from the ellipse fitting
	x, y: length-N numpy arrays
		Recorded position from the ellipse fitting in mm
	min_walk_spd: float
		Lower bound of speed defining a stopped fly in mm/s
	fps: float
		Recording rate in frames per second.
	v_smooth_window: odd int
		Savgol window over which to smooth velocity to calculate heading
	min_walk_dur: float
		Minimum duration to count as walk, in seconds
		
	Returns
	-------

	theta: length-N numpy array
		Corrected orientation modded between [0, 360]
	theta_uncut: length-N numpy array
		Corrected orientation, not modded to [0, 360]
	"""

	theta = np.empty(len(x))
	theta[:] = np.nan
	theta_uncut = np.empty(len(x))
	theta_uncut[:] = np.nan

	# If too short to properly smooth, forget it
	if len(x) < v_smooth_window:
		return theta, theta_uncut

	# Heading calculated using velocity vector, smooth coarsely
	try:
		vx = savgol_filter(x, v_smooth_window, 4, deriv=1, delta=1/fps)
		vy = savgol_filter(y, v_smooth_window, 4, deriv=1, delta=1/fps)
	except np.linalg.LinAlgError:
		vx = np.empty(len(x))*np.nan
		vy = np.empty(len(y))*np.nan
		
	heading = np.arctan2(vy, vx)*180/np.pi % 360
	v = (vx**2 + vy**2)**0.5

	# Split array into separate non-nan sequences
	split_idxs = get_non_nan_splits(v*heading)
	
	for idxs in split_idxs:

		walk_idxs = np.intersect1d(np.where(v > min_walk_spd)[0], idxs)
		stop_idxs = np.intersect1d(np.where(v <= min_walk_spd)[0], idxs)
		if len(walk_idxs) == 0:
			continue
		
		# Walks: compare heading to theta; flip if off by more than 1 quadrant
		walk_rngs = get_consecutive_lists(walk_idxs)
		for rng in walk_rngs:
			
			if len(rng) < fps*min_walk_dur:
				stop_idxs = np.append(stop_idxs, rng)
				continue
			
			for iH in rng:

				# To temporarily hold correctly modded angles
				_theta = theta_raw[iH]
				_heading = heading[iH]

				# Wind 360 deg if heading and angle span a branch cut
				if _heading - _theta > 180:
					_theta += 360
				elif _heading - _theta < -180:
					_theta -= 360
				
				# Flip fly orientation if angles differ by 1 quadrant
				if (abs(_heading - _theta) > 90):
					theta[iH] = (_theta - 180) % 360
				else:
					theta[iH] = _theta % 360

		# Stops: use stop endpoints to guess orientation
		stop_rngs = get_consecutive_lists(np.sort(stop_idxs))
		for rng in [i for i in stop_rngs if len(i) != 0]:
			
			stop_beg = max(rng[0] - 1, idxs[0])
			stop_end = min(rng[-1] + 1, idxs[-1])
			
			# Average theta of stop endpoints, casted to [-180, 180]
			if np.isfinite(theta[stop_beg]*theta[stop_end]):
				_x = np.mean(np.cos(theta[[stop_beg, stop_end]]*np.pi/180))
				_y = np.mean(np.sin(theta[[stop_beg, stop_end]]*np.pi/180))
				avg_theta_in_rng = np.arctan2(_y, _x)*180/np.pi
			else:
				if np.isfinite(theta[stop_beg]):
					avg_theta_in_rng = theta[stop_beg]
				else:
					avg_theta_in_rng = theta[stop_end]
			
			# To hold correctly modded fly orientations during stop
			_theta = np.zeros(len(rng))
			_theta[:] = theta_raw[rng]
			
			# Move branch cut to opposite side of angles
			if (avg_theta_in_rng > 90) + (avg_theta_in_rng < -90):
				if avg_theta_in_rng < 0:
					avg_theta_in_rng += 360
			else:
				_theta[theta_raw[rng] > 180] -= 360
			
			# Flip for which heading and theta are off by 1 quadrant
			head_theta_diff = abs(avg_theta_in_rng - _theta)
			keep_idxs = np.where(head_theta_diff <= 90)[0] + rng[0]
			flip_idxs = np.where(head_theta_diff > 90)[0] + rng[0]
			theta[keep_idxs] = theta_raw[keep_idxs]
			theta[flip_idxs] = theta_raw[flip_idxs] - 180 
			theta[rng] = theta[rng] % 360
		
		# Make an orientation that respects winding number -- not modded
		theta_uncut[idxs[0]] = theta[idxs[0]]
		for iH in range(idxs[0] + 1, idxs[-1]):
			dtheta = theta[iH] - theta[iH - 1]
			if abs(dtheta) < 180:
				theta_uncut[iH] = theta_uncut[iH - 1] + dtheta
			elif dtheta <= -180:
				theta_uncut[iH] = theta_uncut[iH - 1] + dtheta + 360
			elif dtheta >= 180:
				theta_uncut[iH] = theta_uncut[iH - 1] + dtheta - 360
				
	return theta, theta_uncut

def opt_ellipse_orientation(theta_raw, x, y, min_walk_spds, fps, 
							v_smooth_windows, min_walk_dur=0.3):
	"""
	Fit the ellipse orientation by iterating over various heading smoothing 
	windows and walk speed thresholds. Finds the fit such that the 
	orientation flips the least.
	
	Args
	-------

	theta_raw: length-N numpy array
		Recorded orientation from the ellipse fitting
	x, y: length-N numpy arrays
		Recorded position from the ellipse fitting in mm
	min_walk_spds: list of floats
		Speed thresholds defining a stopped fly in mm/s
	fps: float
		Recording rate in frames per second.
	v_smooth_windows: list of odd ints
		Savgol window lengths over which to smooth velocity to 
		calculate heading
	min_walk_dur: float
		Minimum duration to count as walk, in seconds
		
	Returns
	-------

	theta: length-N numpy array
		Optimal orientation, modded between [0, 360], compared over all 
		of the walk thresholds and smoothing window sizes.
	theta_uncut: length-N numpy array
		Corrected orientation, not modded to [0, 360]
	"""
	
	# Find an optimal spd threshold which minimizes orientation flips.
	errors = np.zeros((len(min_walk_spds), len(v_smooth_windows)))
	for iS, min_walk_spd in enumerate(min_walk_spds):
		for iW, v_smooth_window in enumerate(v_smooth_windows):
			_, _theta_uncut = fix_ellipse_orientation(theta_raw, x, y,
								min_walk_spd, fps, int(v_smooth_window), 
								min_walk_dur)
			num_flips = np.sum(abs(np.diff(_theta_uncut)) > 120)
			errors[iS, iW] = num_flips
			
	iS_min, iW_min = np.unravel_index(errors.argmin(), errors.shape)
	theta, theta_uncut = fix_ellipse_orientation(theta_raw, x, y,
							min_walk_spds[iS_min], fps, 
							int(v_smooth_windows[iW_min]),
							min_walk_dur)
	
	return theta, theta_uncut
