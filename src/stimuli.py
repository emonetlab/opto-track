"""
Functions for projecting distinct stimuli

Created by Nirag Kadakia at 15:12 06-27-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
from utils import convert_units
from load_save_data import load_settings, get_stim_vids_dir, \
						   load_stim_video_metadata


class stim_protocol():
	"""
	General class for a stimulus protocol. 
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool;
		If True, background is red and stimuli are black.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		"""
		
		self.win = win
		self.rec_rate = rec_rate
		self.max_flies = max_flies
		self.invert = invert
		if invert == True:
			self.shapes_color = 'black'
			if framepack == True:
				self.win.color = 'white'
			else:
				self.win.color = 'red'
		else:
			if framepack == True:
				self.shapes_color = 'white'
			else:
				self.shapes_color = 'red'
			self.win.color = 'black'
			
	def initialize_shapes(self):
		"""
		Initialize the psychopy shapes. This takes time, so must be done
		once up front. Then Attrs of shapes are updated frame-by-frame 
		using the self.update method.
		"""
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=1, elementMask='circle',
						xys=np.zeros((1, 2)), 
						sizes=[0.1, 0.2],
						colors=self.shapes_color, elementTex=None)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update stimulus at each frame. This may depend on detected flies, 
		stored in objects, or other aspects, stored in the centroidTracker 
		object ct. It can also depend on the frame number.
		
		Args
		-------
		
		ct: centroidTracker object instance
			To record flies 
		centroids: (self.max_flies x 2) array.
			Center positions of all possible flies, by ID
		states: (self.max_flies x 2) array
			States of flies; 1: detected; 2: non-detected
		frm_num: int
			Video frame number.
		"""
		
		self.shapes.xys = np.random.uniform(-1, 1, (1, 2))


class static_ribbon(stim_protocol):
	"""
	Single static ribbon horizontal to wind flow.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool
		If True, background is red and stimuli are black.
	width: float
		width of ribbon in mm.
	locs: list of floats
		Y-placements of ribbons in stimulus units (-1 to 1)
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, intensity=35,
				 width=2, locs=[0, -0.5, 0.5]):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		width: float
			Width of ribbon in mm
		locs: list of floats
			Y-placements of ribbons in stimulus units (-1 to 1)
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		conv_cam_to_proj = convert_units().conv_cam_to_proj
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		stim_units_per_px = (conv_cam_to_proj(0, 1)[1] - 
							conv_cam_to_proj(0, 0)[1])
		
		self.framepack_num = int(2*framepack + 1)
		self.intensity = intensity
		self.width = width
		self.width_stim_units = width/mm_per_px*abs(stim_units_per_px)
		self.locs = locs
		
	def initialize_shapes(self):
		
		from psychopy import visual
		xys = np.vstack((np.zeros(len(self.locs)), self.locs)).T
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=len(self.locs), fieldShape='square',
						elementMask=None, xys=xys,
						sizes=[2, self.width_stim_units], 
						colorSpace='rgb255', colors=(0,0,0),
						elementTex=None)

		if self.framepack_num == 3:
			self.shapes.colors = (self.intensity, self.intensity, self.intensity)
		else:
			self.shapes.colors = (self.intensity, 0, 0)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Nothing to update since static.
		"""
		
		pass


class box_on_fly(stim_protocol):
	"""
	Red circle illuminates the fly at all times.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	conv_cam_to_proj: conv_cam_to_proj method of convert_units class
		Converts units from camera pixels to stimulus units.
	ellipse_dx: floa
		Horizontal size of circle in stimulus units.
	ellipse_dy: float
		Vertical size of circle in stimulus units.
	"""
		
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, circle_diam=10):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		circle_diam: float
			Diameter of stimulus circle covering each fly, in mm.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		self.config = load_settings()
		self.conv_cam_to_proj = convert_units().conv_cam_to_proj
		mm_per_px = self.config.getfloat("Camera", "mm_per_px")
		stim_dx_per_px = (self.conv_cam_to_proj(1, 0)[0] - 
							self.conv_cam_to_proj(0, 0)[0])
		stim_dy_per_px = (self.conv_cam_to_proj(0, 1)[1] - 
							self.conv_cam_to_proj(0, 0)[1])
		self.ellipse_dx = circle_diam/mm_per_px*stim_dx_per_px
		self.ellipse_dy = circle_diam/mm_per_px*stim_dy_per_px
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		each fly. Each box is red; just moved off screen if fly disappears.
		"""
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.max_flies, elementMask='circle',
						xys=np.zeros((self.max_flies, 2)), 
						sizes=[self.ellipse_dx, self.ellipse_dy],
						colors=self.shapes_color, elementTex=None)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update boxes to positions of moved flies.
		"""
		
		xs, ys = self.conv_cam_to_proj(centroids[:, 0], centroids[:, 1])
		
		# The 3.*(1. - states)*3 term just pushes the boxes off the screen,
		# without having to change their color, which is time-consuming
		xys = [xs + 3*(1 - states), ys]
		
		# Transpose list to (Nx2); faster than vstacking as numpy array
		vals = list(map(list, zip(*xys)))
		self.shapes.xys = vals
		
		
class horiz_moving_bars(stim_protocol):
	"""
	Horizontal bar moving lateral to wind direction.
	
	Attrs
	-------

	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	on_width: float
		Width of bar in mm.
	off_width: float
		Width of region between bar in mm. If None, set to on_width
	on_width_stim_units: float
		Width of bar in stimulus units.
	off_width_stim_units: float
		Width of space between bars in stimulus units.
	num_bars: int
		Number of total bar shapes in psychopy object
	speed: float
		Speed in mm/s
	proj_speed: float
		Speed of bar in stimulus units per frame.
	invert: bool
		Bars are black; background is red
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	xs: List of length num_bars.
		Positions of the bars.
	ys: list of length num_bars.
		Positions of the bars.
	On_T, Off_T: floats
		If both are not None, then bars are presented in blocks of
		On_T seconds, interrupted by Off_T seconds of no stimulus.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, intensity=255,
				 framepack=False, speed=1, on_width=2, off_width=None,
				 On_T=None, Off_T=None):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack_num: int
			3 if packing RGB frames; 1 otherwise
		speed: float
			Speed of bar in mm/s.
		on_width: float
			Width of bar in mm.
		off_width: float
			Width of region between bar in mm. If None, set to on_width
		invert: bool
			Bars are black; background is red
		On_T, Off_T: floats
			If both are not None, then bars are presented in blocks of
			On_T seconds, interrupted by Off_T seconds of no stimulus.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		conv_cam_to_proj = convert_units().conv_cam_to_proj
		stim_units_per_px = (conv_cam_to_proj(0, 1)[1] - 
							 conv_cam_to_proj(0, 0)[1])
		
		if off_width is None:
			off_width = on_width
		self.on_width = on_width
		self.off_width = off_width
		self.on_width_stim_units = on_width/mm_per_px*abs(stim_units_per_px)
		self.off_width_stim_units = off_width/mm_per_px*abs(stim_units_per_px)
		self.intensity = intensity
		self.speed = speed
		self.proj_speed = speed/self.rec_rate/mm_per_px*abs(stim_units_per_px)
		
		self.framepack_num = int(2*framepack + 1)
		
		if (On_T is not None) and (Off_T is not None):
			self.On_T = On_T
			self.Off_T = Off_T
			self.num_On_Off_frms = int((Off_T + On_T)*\
									 rec_rate*self.framepack_num)
			self.num_On_frms = int(On_T*rec_rate*self.framepack_num)
		else:
			self.On_T = None
			self.Off_T = None

	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Bars evenly spaced on 
		screen from top to bottom.
		"""
		
		# Width in stimulus units of 1 cycle
		self.cycle_width = self.on_width_stim_units + self.off_width_stim_units
		
		# Min # of whiff + blank cycles needed to fully cover arena
		cycles_per_arena = 2//self.cycle_width + 1
		
		# Full width of pattern area -- covers space equal to 3 arenas
		self.full_width = 3*cycles_per_arena*self.cycle_width
		self.ys = np.arange(-self.full_width/2., self.full_width/2., 
					self.cycle_width)
		self.num_bars = len(self.ys)
		self.xs = np.zeros(self.num_bars)
		
		xys = [self.xs, self.ys]
		xys = list(map(list, zip(*xys)))
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.num_bars, elementMask=None,
						xys=xys, fieldShape='sqr', 
						sizes=[2, self.on_width_stim_units],
						colorSpace='rgb255', colors=(0,0,0),
						elementTex=None)

		if self.framepack_num == 3:
			self.shapes.colors = (self.intensity, self.intensity, self.intensity)
		else:
			self.shapes.colors = (self.intensity, 0, 0)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update positions to move by one step; cycle at boundaries.
		"""
		
		self.xs = np.zeros(self.num_bars)
		self.ys += self.proj_speed/self.framepack_num
		
		# If using blocks, then push bars off screen during blank period
		if (self.On_T is not None) and (self.Off_T is not None):
			block_frm = np.mod(frm_num, self.num_On_Off_frms)
			if (block_frm > self.num_On_frms):
				self.xs = np.ones(self.num_bars)*3
		
		xys = [self.xs, self.ys]
		xys = list(map(list, zip(*xys)))
		self.ys[self.ys > self.full_width/2.] = -self.full_width/2
		self.ys[self.ys < -self.full_width/2.] = self.full_width/2.
		self.shapes.xys = xys
		
		
class horiz_moving_bars_single_edge(stim_protocol):
	"""
	Horizontal bar moving lateral to wind direction, where the bar flashes
	on the entire fly first, then moves off. This gives only an off edge.
	Or: can do flashes in front of the fly then moves on. This gives only
	an on edge.
	
	Attrs
	-------

	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	conv_cam_to_proj: conv_cam_to_proj method of convert_units class
		Converts units from camera pixels to stimulus units.
	width: float
		Width of bar in mm.
	width_stim_units: float
		Width of bar in stimulus units.
	speed: float
		Speed in mm/s
	proj_speed: float
		Speed of bar in stimulus units per frame.
	invert: bool
		Bars are black; background is red
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	length_stim_units: float
		Length of stimulus lateral to motion direction, in stimulus units 
		(-1 to 1)
	edge_type: str
		`on' or `off'
	on_edge_shift_stim_units: positive float
		shift ahead of fly for on edge, in stimulus units
	switch_T: positive float
		Length of time between whiffs
	motion_T: positive float, less than switch T
		Length of the whiff
	num_cycle_frms: int
		Number of frames per cycle
	num_motion_frms: int
		Number of frames for scintillator
	cycle_frms: list of length of the total recording frames
		Contains the frame number within the cycle
	xs, ys: lists of length num_bars.
		Positions of the bars.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, intensity=255, 
				 framepack=False, speed=1, width=2, length_stim_units=0.35,
				 edge_type='off', on_edge_shift=3, switch_T=4, motion_T=2):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack_num: int
			3 if packing RGB frames; 1 otherwise
		speed: float
			Speed of bar in mm/s.
		width: float
			Width of bar in mm.
		invert: bool
			Bars are black; background is red
		length_stim_units: float
			Length of stimulus lateral to motion direction, in stimulus units 
			(-1 to 1)
		edge_type: str
			`on' or `off'
		on_edge_shift: positive float
			shift ahead of fly for on edge, in mm
		switch_T: float
			Time in seconds between whiffs
		motion_T: float
			Length in seconds of whiff.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		self.conv_cam_to_proj = convert_units().conv_cam_to_proj
		stim_units_per_px = (self.conv_cam_to_proj(0, 1)[1] - 
							 self.conv_cam_to_proj(0, 0)[1])
		self.width = width
		self.width_stim_units = width/mm_per_px*abs(stim_units_per_px)
		self.length_stim_units = length_stim_units
		self.speed = speed
		self.proj_speed = speed/self.rec_rate/mm_per_px*abs(stim_units_per_px)
		self.framepack_num = int(2*framepack + 1)
		
		self.edge_type = edge_type
		self.on_edge_shift_stim_units = on_edge_shift/mm_per_px*\
										  abs(stim_units_per_px)
		self.switch_T = switch_T
		self.motion_T = motion_T
		self.num_cycle_frms = int(switch_T*self.rec_rate*self.framepack_num)
		self.num_motion_frms = int(motion_T*self.rec_rate*self.framepack_num)
		self.cycle_frms = []
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Bars evenly spaced on 
		screen from top to bottom.
		"""
		
		self.xs = np.zeros(self.max_flies)
		self.ys = np.zeros(self.max_flies)
		xys = [self.xs, self.ys]
		xys = list(map(list, zip(*xys)))
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.max_flies, elementMask=None,
						xys=xys, fieldShape='sqr', 
						sizes=[self.length_stim_units, self.width_stim_units],
						colorSpace='rgb255', colors=(0,0,0),
						elementTex=None)

		if self.framepack_num == 3:
			self.shapes.colors = (self.intensity, self.intensity, self.intensity)
		else:
			self.shapes.colors = (self.intensity, 0, 0)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update positions to move by one step; cycle at boundaries.
		"""
		
		cycle_frm = np.mod(frm_num, self.num_cycle_frms)
		if self.edge_type == 'off':
			
			# At beginning of cycle, flash bar onto top of fly, else: shift
			if cycle_frm == 0:
				self.xs, self.ys = self.conv_cam_to_proj(centroids[:, 0], 
							   centroids[:, 1])
			else:
				self.ys += self.proj_speed/self.framepack_num
		
		elif self.edge_type == 'on':
			
			# At beginning of cycle, flash bar in front of fly; else: shift
			if cycle_frm == 0:
				self.xs, self.ys = self.conv_cam_to_proj(centroids[:, 0], 
							   centroids[:, 1])
				self.ys -= self.on_edge_shift_stim_units
			else:
				self.ys += self.proj_speed/self.framepack_num
			
		# After motion cycle, no bars
		if cycle_frm > self.num_motion_frms:
			self.xs += 3
		
		# Move missing flies off screen
		xys = [self.xs + 3*(1 - states), self.ys]
		xys = list(map(list, zip(*xys)))
		self.shapes.xys = xys
		
		if frm_num > 0:
			self.cycle_frms.append(cycle_frm)
		
class vert_moving_bars(stim_protocol):
	"""
	Vertical bar moving parallel/anti-parallel to wind direction.
	
	Attrs
	-------

	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	on_width: float
		Width of bar in mm.
	off_width: float
		Width of region between bar in mm. If None, set to on_width
	on_width_stim_units: float
		Width of bar in stimulus units.
	off_width_stim_units: float
		Width of region between bars in stimulus units.
	num_bars: int
		Number of to have concurrently (evenly spaced).
	speed: float
		Speed in mm/s
	proj_speed: float
		Speed of bar in stimulus units per frame.
	invert: bool
		Bars are black; background is red
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	xs: list of length num_bars.
		Positions of the bars.
	On_T, Off_T: floats
		If both are not None, then bars are presented in blocks of
		On_T seconds, interrupted by Off_T seconds of no stimulus.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, intensity=255,
				 framepack=False, speed=1, on_width=2, off_width=None, 
				 On_T=None, Off_T=None):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		speed: float
			Speed of bar in mm/s.
		on_width: float
			Width of bar in mm.
		off_width: float
			Width of region between bar in mm. If None, set to on_width
		invert: bool
			Bars are black; background is red
		On_T, Off_T: floats
			If both are not None, then bars are presented in blocks of
			On_T seconds, interrupted by Off_T seconds of no stimulus.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		conv_cam_to_proj = convert_units().conv_cam_to_proj
		stim_units_per_px = (conv_cam_to_proj(1, 0)[0] - 
							 conv_cam_to_proj(0, 0)[0])
		
		if off_width is None:
			off_width = on_width
		self.on_width = on_width
		self.off_width = off_width
		self.on_width_stim_units = on_width/mm_per_px*abs(stim_units_per_px)
		self.off_width_stim_units = off_width/mm_per_px*abs(stim_units_per_px)
		
		self.intensity = intensity
		
		self.speed = speed
		self.proj_speed = speed/self.rec_rate/mm_per_px*abs(stim_units_per_px)
		
		self.framepack_num = int(2*framepack + 1)
		
		if (On_T is not None) and (Off_T is not None):
			self.On_T = On_T
			self.Off_T = Off_T
			self.num_On_Off_frms = int((Off_T + On_T)*\
									 rec_rate*self.framepack_num)
			self.num_On_frms = int(On_T*rec_rate*self.framepack_num)
		else:
			self.On_T = None
			self.Off_T = None
			
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Bars evenly spaced on 
		screen from top to bottom.
		"""
		
		# Width in stimulus units of 1 cycle
		self.cycle_width = self.on_width_stim_units + self.off_width_stim_units
		
		# Min # of whiff + blank cycles needed to fully cover arena
		cycles_per_arena = 2//self.cycle_width + 1
		
		# Full width of pattern area -- covers space equal to 3 arenas
		self.full_width = 3*cycles_per_arena*self.cycle_width
		self.xs = np.arange(-self.full_width/2., self.full_width/2., 
					self.cycle_width)
		self.num_bars = len(self.xs)
		self.ys = np.zeros(self.num_bars)
		
		xys = [self.xs, self.ys]
		xys = list(map(list, zip(*xys)))
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.num_bars, elementMask=None,
						xys=xys, fieldShape='sqr',
						sizes=[self.on_width_stim_units, 2],
						colorSpace='rgb255', colors=(0,0,0),
						elementTex=None)

		if self.framepack_num == 3:
			self.shapes.colors = (self.intensity, self.intensity, self.intensity)
		else:
			self.shapes.colors = (self.intensity, 0, 0)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update positions to move by one step; cycle at boundaries.
		"""

		self.xs += self.proj_speed/self.framepack_num
		self.ys = np.zeros(self.num_bars)
		
		# If using blocks, then push bars off screen during blank period
		if (self.On_T is not None) and (self.Off_T is not None):
			block_frm = np.mod(frm_num, self.num_On_Off_frms)
			if (block_frm > self.num_On_frms):
				self.ys = np.ones(self.num_bars)*3
		xys = [self.xs, self.ys]
		xys = list(map(list, zip(*xys)))
		self.xs[self.xs > self.full_width/2.] = -self.full_width/2
		self.xs[self.xs < -self.full_width/2.] = self.full_width/2.
		self.shapes.xys = xys

		
class vert_moving_bars_single_edge(stim_protocol):
	"""
	Horizontal bar moving lateral to wind direction.
	
	Attrs
	-------

	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	width: float
		Width of bar in mm.
	width_stim_units: float
		Width of bar in stimulus units.
	speed: float
		Speed in mm/s
	proj_speed: float
		Speed of bar in stimulus units per frame.
	invert: bool
		Bars are black; background is red
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	length_stim_units: float
		Length of stimulus lateral to motion direction, in stimulus units
		(-1 to 1)
	edge_type: str
		`on' or `off'
	on_edge_shift_stim_units: positive float
		shift ahead of fly for on edge, in stimulus units
	switch_T: positive float
		Length of time between whiffs
	motion_T: positive float, less than switch T
		Length of the whiff
	num_cycle_frms: int
		Number of frames per cycle
	num_motion_frms: int
		Number of frames for scintillator
	cycle_frms: list of length of the total recording frames
		Contains the frame number within the cycle
	xs, ys: lists of length num_bars.
		Positions of the bars.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, speed=1, width=2, length_stim_units=0.35,
				 edge_type='off', on_edge_shift=3, switch_T=4, motion_T=2):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		speed: float
			Speed of bar in mm/s.
		width: float
			Width of bar in mm.
		invert: bool
			Bars are black; background is red
		length_stim_units: float
			Length of stimulus lateral to motion direction, in stimulus units 
			(-1 to 1)
		edge_type: str
			`on' or `off'
		on_edge_shift: positive float
			shift ahead of fly for on edge, in mm
		switch_T: float
			Time in seconds between whiffs
		motion_T: float
			Length in seconds of whiff.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		self.conv_cam_to_proj = convert_units().conv_cam_to_proj
		stim_units_per_px = (self.conv_cam_to_proj(1, 0)[0] - 
							 self.conv_cam_to_proj(0, 0)[0])
		self.width = width
		self.width_stim_units = width/mm_per_px*abs(stim_units_per_px)
		self.length_stim_units = length_stim_units
		self.speed = speed
		self.proj_speed = speed/self.rec_rate/mm_per_px*abs(stim_units_per_px)
		self.framepack_num = int(2*framepack + 1)
		
		self.edge_type = edge_type
		self.on_edge_shift_stim_units = on_edge_shift/mm_per_px*\
										  abs(stim_units_per_px)
		self.switch_T = switch_T
		self.motion_T = motion_T
		self.num_cycle_frms = int(switch_T*self.rec_rate*self.framepack_num)
		self.num_motion_frms = int(motion_T*self.rec_rate*self.framepack_num)
		self.cycle_frms = []
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Bars evenly spaced on 
		screen from top to bottom.
		"""
		
		self.xs = np.zeros(self.max_flies)
		self.ys = np.zeros(self.max_flies)
		xys = [self.xs, self.ys]
		xys = list(map(list, zip(*xys)))
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.max_flies, elementMask=None,
						xys=xys, fieldShape='sqr', 
						sizes=[self.width_stim_units, self.length_stim_units],
						colors=self.shapes_color, elementTex=None)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update positions to move by one step; cycle at boundaries.
		"""
		
		cycle_frm = np.mod(frm_num, self.num_cycle_frms)
		if self.edge_type == 'off':
			
			# At beginning of cycle, flash bar onto top of fly, else: shift
			if cycle_frm == 0:
				self.xs, self.ys = self.conv_cam_to_proj(centroids[:, 0], 
							   centroids[:, 1])
			else:
				self.xs += self.proj_speed/self.framepack_num
		
		elif self.edge_type == 'on':
			
			# At beginning of cycle, flash bar in front of fly; else: shift
			if cycle_frm == 0:
				self.xs, self.ys = self.conv_cam_to_proj(centroids[:, 0], 
							   centroids[:, 1])
				self.xs -= self.on_edge_shift_stim_units
			else:
				self.xs += self.proj_speed/self.framepack_num
			
		# After motion cycle, no bars
		if cycle_frm > self.num_motion_frms:
			self.xs += 3
		
		# Move missing flies off screen
		xys = [self.xs + 3*(1 - states), self.ys]
		xys = list(map(list, zip(*xys)))
		self.shapes.xys = xys
		
		if frm_num > 0:
			self.cycle_frms.append(cycle_frm)


class full_field_flash(stim_protocol):
	"""
	Flash entire arena at a given frequency and intensity. 
	Flashes occur at defined frequency and duration, or 
	alternatively frequency and duty cycle. In addition, 
	flashes occur in `blocks", the length and period 
	of which are separately defined:
	
	
				  <--ON block-->       <-OFF block->    <---ON block--> 
	Odor signal: |-|__|-|__|-|__|-|____________________|-|__|-|__|-|__|-|
				 __    
				 Flash duration
	
	Attrs
	--------
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack: bool
		If True, pack 3 frames into single 16.67 ms flip
	freq: float
		Frequency of flashes in frames/sec. Bounded below by the 
		recording rate. 
	dur: float
		Duration of flash in seconds. If duty_cycle is not None, then 
		this kwarg is ignored and duty_cycle and freq are used to calculate
		the flash duration.
	duty_cycle: float <= 1, or None
		If None, ignored and dur and freq used to calculate. Otherwise,
		percentage of time the flash is on
	num_flash_cycle_frms: int
		Number of frames between start of each flash (ie period of flashing)
	num_flash_frms: int
		Number of frames for an invididual flash
	On_T: float
		Length of block in seconds during which flashes are delivered
	Off_T: float
		Length of block in seconds during which no flashes are delivered. 
		This is cycled with On_T to give a total block period of On_T+Off_T
	num_On_Off_frms: int
		Number of frames between start of each stimulus block (ie period 
		of blocks)
	num_On_frms: int
		Number of frames for each ON part of the cycle (ie number of frames
		during which flashes will occur)
	intensity: int
		The rgb255 intensity for the red light displayed on the projector.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, freq=1, dur=0.5, duty_cycle=None, 
				 On_T=5.0, Off_T=5.0, intensity=255):
		
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid.
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		freq: float
			Frequency of flashes in frames/sec. Bounded below by the 
			recording rate. 
		dur: float
			Duration of flash in seconds. If duty_cycle is not None, then 
			this kwarg is ignored and duty_cycle and freq are used to calculate
			the flash duration.
		duty_cycle: float <= 1, or None
			If None, ignored and dur and freq used to calculate. Otherwise,
			percentage of time the flash is on
		On_T: float
			Length of block in seconds during which flashes are delivered
		Off_T: float
			Length of block in seconds during which no flashes are delivered. 
			This is cycled with On_T to give a total block period of On_T+Off_T
		intensity: int
			The rgb255 intensity for the red light displayed on the projector.
		"""

		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		self.framepack_num = int(2*framepack + 1)
		self.freq = freq
		if duty_cycle is None:
			self.dur = dur
			self.duty_cycle = dur*freq
			assert self.duty_cycle <= 1, 'Duty cycle (dur*freq) must be	'\
				'less than 1. Decrease either dur or freq, or pass duty_cycle '\
				'in lieu of dur.'
		else:
			assert duty_cycle <= 1, "duty_cycle must be less than 1. '\
				'Decrease or pass as `None'."
			print("Duty cycle set at %s. Ignoring kwarg `dur'" % duty_cycle)
			self.duty_cycle = duty_cycle
			self.dur = duty_cycle/freq
		assert self.rec_rate >= 2*freq, \
			'Flash rate cannot be greater than half the recording rate.'
		assert intensity <= 255 and intensity >= 0, 'Intensity must be '\
			'greater than or equal to 0 and less than or equal 255.'
		
		self.num_flash_cycle_frms = int(self.rec_rate*self.framepack_num/freq)
		self.num_flash_frms = int(self.num_flash_cycle_frms*self.duty_cycle)
		
		# Sum of On and Off gives full cycle
		self.On_T = On_T
		self.Off_T = Off_T
		self.num_On_Off_frms = int((Off_T + On_T)*rec_rate*self.framepack_num)
		self.num_On_frms = int(On_T*rec_rate*self.framepack_num)

		# Set intensity of flash
		self.intensity = intensity
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		whole screen.
		"""
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=1, elementMask=None,
						xys=[[0, 0]], fieldShape='sqr',
						sizes=[[2, 2]], colorSpace='rgb255',
						colors=(0,0,0), elementTex=None)

		if self.framepack_num == 3:
			self.shapes.colors = (self.intensity, self.intensity, self.intensity)
		else:
			self.shapes.colors = (self.intensity, 0, 0)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update position if on frame number to update. Make black by moving box
		too far to right, off screen. 
		"""
		
		# Frame in each On/Off block cycle
		block_frm = np.mod(frm_num, self.num_On_Off_frms)
		
		# Frame in each flash cycle
		flash_frm = np.mod(frm_num, self.num_flash_cycle_frms)
		
		# If within both in flash and ON block, then there is a flash
		if (block_frm < self.num_On_frms)*(flash_frm < self.num_flash_frms):
			self.shapes.xys = [[0, 0]]
		else:
			self.shapes.xys = [[3, 0]]


class full_field_flash_sigmoid(stim_protocol):
	"""
	Flash entire arena at a given frequency and intensity with the 
	front edge of the flash increasing as a sigmoid. A frequency
	and duration of the flashes is needed, as well as the slope of 
	the front edge of the sigmoid. These flashes occur in blocks, with 
	ON and OFF blocks also defined by the user. Note that in contrast
	to `full_field_flash,' the blank comes at the beginning of the block
	cyle, then comes the flash following that. 
	
	Attrs
	--------
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	framepack: bool
		If True, pack 3 frames into single 16.67 ms flip
	freq: float
		Frequency of flashes in frames/sec. Bounded below by the 
		recording rate. 
	dur: float
		Duration of flash in seconds. If duty_cycle is not None, then 
		this kwarg is ignored and duty_cycle and freq are used to calculate
		the flash duration.
	num_flash_cycle_frms: int
		Number of frames between start of each flash (ie period of flashing)
	num_flash_frms: int
		Number of frames for an invididual flash
	flash_start: int
		The frame at which the sigmoid should reach half-max, also considered
		the start of the flash duration.
	On_T: float
		Length of block in seconds during which flashes are delivered
	Off_T: float
		Length of block in seconds during which no flashes are delivered. 
		This is cycled with On_T to give a total block period of On_T+Off_T
	num_On_Off_frms: int
		Number of frames between start of each stimulus block (ie period 
		of blocks)
	num_On_frms: int
		Number of frames for each ON part of the cycle (ie number of frames
		during which flashes will occur)
	intensity: int
		The rgb255 intensity for the red light displayed on the projector.
	slope: float
		The slope of the sigmoid curve. The greater the slope, the more 
		like a straight edge.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	"""
	
	def __init__(self, win, rec_rate, max_flies, framepack=False, invert=False,
		  freq=1, dur=0.5, On_T=5.0, Off_T=5.0, intensity=255, slope = 0.05):
		
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		freq: float
			Frequency of flashes in frames/sec. Bounded below by the 
			recording rate. 
		dur: float
			Duration of flash in seconds. If duty_cycle is not None, then 
			this kwarg is ignored and duty_cycle and freq are used to calculate
			the flash duration.
		On_T: float
			Length of block in seconds during which flashes are delivered
		Off_T: float
			Length of block in seconds during which no flashes are delivered. 
			This is cycled with On_T to give a total block period of On_T+Off_T
		intensity: int
			The rgb255 intensity for the red light displayed on the projector.
		slope: float
			The slope of the sigmoid curve.
		"""

		super().__init__(win, rec_rate, max_flies, invert=False, 
						 framepack=framepack)
		
		self.framepack_num = int(2*framepack + 1)
		self.freq = freq
		self.dur = dur
		self.slope = slope
		assert self.rec_rate >= 2*freq, \
		  'Flash rate cannot be greater than half the recording rate.'
		assert self.slope >= 0.026, 'Slopes cannot be less than than 0.026 ' \
		  'as that slope is too shallow for a proper sigmoid'
		assert self.freq*self.dur <= 1, 'Duty cycle (dur*freq) must be	'\
		  'less than 1. Decrease either dur or freq, or pass duty_cycle '\
		  'in lieu of dur.'
		
		# Tracking the duration and period of the stimuli
		self.num_flash_frms = int(self.dur*self.framepack_num*self.rec_rate)
		self.num_flash_cycle_frms = int(self.framepack_num*self.rec_rate\
			/self.freq)
			
		# The flash starts 
		self.flash_start = self.num_flash_cycle_frms - self.num_flash_frms
		
		# Sum of On and Off gives full cycle
		self.On_T = On_T
		self.Off_T = Off_T
		self.num_On_Off_frms = int((Off_T + On_T)*rec_rate*self.framepack_num)
		self.num_On_frms = int(On_T*rec_rate*self.framepack_num)

		# Set intensity of flash
		self.intensity = intensity
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		whole screen.
		"""
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=1, elementMask=None,
						xys=[[0, 0]], fieldShape='sqr',
						sizes=[[2, 2]], colorSpace = 'rgb255',
						elementTex=None, colors = (0, 0, 0))
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update position if on frame number to update. Make black by moving box
		too far to right, off screen. 
		"""
		
		# Frame in each On/Off block cycle
		block_frm = np.mod(frm_num, self.num_On_Off_frms)
		
		# Frame in each flash cycle
		flash_frm = np.mod(frm_num, self.num_flash_cycle_frms)
		
		# Check for On block and that flash is not cut off by end of block
		if (block_frm < self.num_On_frms + self.num_flash_frms):

			# Value of intensity at each frame within each period. The 
			# `flash_frm' increases linearly from 0 to max through one
			# flash + blank period
			rgb_value = np.floor(self.intensity/(1 + np.exp(-self.slope\
			  *(flash_frm - self.flash_start))))

			# Framepacking check for intensities
			if self.framepack_num == 3:
				self.shapes.colors = (rgb_value, rgb_value, rgb_value)
			else:
				self.shapes.colors = (rgb_value, 0, 0)
		else:
			self.shapes.colors = (0, 0, 0)
			

class full_field_flash_sigmoid_linearized(stim_protocol):
	"""
	Flash entire arena at a given frequency and intensity with the 
	front edge of the flash increasing as a sigmoid. A frequency
	and duration of the flashes is needed, as well as the slope of 
	the front edge of the sigmoid. These flashes occur in blocks, with 
	ON and OFF blocks also defined by the user. Note that in contrast
	to `full_field_flash,' the blank comes at the beginning of the block
	cyle, then comes the flash following that. 
	
	Attrs
	--------
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	framepack: bool
		If True, pack 3 frames into single 16.67 ms flip
	freq: float
		Frequency of flashes in frames/sec. Bounded below by the 
		recording rate. 
	dur: float
		Duration of flash in seconds. If duty_cycle is not None, then 
		this kwarg is ignored and duty_cycle and freq are used to calculate
		the flash duration.
	num_flash_cycle_frms: int
		Number of frames between start of each flash (ie period of flashing)
	num_flash_frms: int
		Number of frames for an invididual flash
	flash_start: int
		The frame at which the sigmoid should reach half-max, also considered
		the start of the flash duration.
	On_T: float
		Length of block in seconds during which flashes are delivered
	Off_T: float
		Length of block in seconds during which no flashes are delivered. 
		This is cycled with On_T to give a total block period of On_T+Off_T
	num_On_Off_frms: int
		Number of frames between start of each stimulus block (ie period 
		of blocks)
	num_On_frms: int
		Number of frames for each ON part of the cycle (ie number of frames
		during which flashes will occur)
	intensity: int
		The rgb255 intensity for the red light displayed on the projector.
	slope: float
		The slope of the sigmoid curve. The greater the slope, the more 
		like a straight edge.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	"""
	
	def __init__(self, win, rec_rate, max_flies, framepack=False, invert=False,
		  freq=1, dur=0.5, On_T=5.0, Off_T=5.0, intensity=35, slope=0.05):
		
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		freq: float
			Frequency of flashes in frames/sec. Bounded below by the 
			recording rate. 
		dur: float
			Duration of flash in seconds. If duty_cycle is not None, then 
			this kwarg is ignored and duty_cycle and freq are used to calculate
			the flash duration.
		On_T: float
			Length of block in seconds during which flashes are delivered
		Off_T: float
			Length of block in seconds during which no flashes are delivered. 
			This is cycled with On_T to give a total block period of On_T+Off_T
		intensity: int
			The rgb255 intensity for the red light displayed on the projector.
		slope: float
			The slope of the sigmoid curve.
		"""

		super().__init__(win, rec_rate, max_flies, invert=False, 
						 framepack=framepack)
		
		self.framepack_num = int(2*framepack + 1)
		self.freq = freq
		self.dur = dur
		self.slope = slope
		assert self.rec_rate >= 2*freq, \
		  'Flash rate cannot be greater than half the recording rate.'
		assert self.slope >= 0.026, 'Slopes cannot be less than than 0.026 ' \
		  'as that slope is too shallow for a proper sigmoid'
		assert self.freq*self.dur <= 1, 'Duty cycle (dur*freq) must be	'\
		  'less than 1. Decrease either dur or freq, or pass duty_cycle '\
		  'in lieu of dur.'
		
		# Tracking the duration and period of the stimuli
		self.num_flash_frms = int(self.dur*self.framepack_num*self.rec_rate)
		self.num_flash_cycle_frms = int(self.framepack_num*self.rec_rate\
			/self.freq)
			
		# The flash starts 
		self.flash_start = self.num_flash_cycle_frms - self.num_flash_frms
		
		# Sum of On and Off gives full cycle
		self.On_T = On_T
		self.Off_T = Off_T
		self.num_On_Off_frms = int((Off_T + On_T)*rec_rate*self.framepack_num)
		self.num_On_frms = int(On_T*rec_rate*self.framepack_num)

		# Linear slope calculation
		self.lin_slope =  (35*self.slope)/4 
		self.x_intercept =  -(-68 + (35*self.flash_start*self.slope))/\
							   (35*self.slope)


		# Set intensity of flash
		self.intensity = intensity
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		whole screen.
		"""
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=1, elementMask=None,
						xys=[[0, 0]], fieldShape='sqr',
						sizes=[[2, 2]], colorSpace = 'rgb255',
						elementTex=None, colors = (0, 0, 0))
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update position if on frame number to update. Make black by moving box
		too far to right, off screen. 
		"""
		
		# Frame in each On/Off block cycle
		block_frm = np.mod(frm_num, self.num_On_Off_frms)
		
		# Frame in each flash cycle
		flash_frm = np.mod(frm_num, self.num_flash_cycle_frms)
		
		# Check for On block and that flash is not cut off by end of block
		if (block_frm < self.num_On_frms + self.num_flash_frms):
			# If the value of the line is greater than 0 and less than 35, 
			# set rgb_value to the value of the line
			if self.lin_slope*(flash_frm - self.flash_start) + 17 >= 0 and \
				self.lin_slope*(flash_frm - self.flash_start) + 17 < 35:
				rgb_value = np.floor(self.lin_slope*(flash_frm - 
				self.flash_start) + 17)
			
			# If greater, make it a ramp
			elif self.lin_slope * (flash_frm - self.flash_start) + 17 >= 35:
				rgb_value = 35
			
			# Otherwise, set to 0
			else:
				rgb_value = 0

			# Framepacking check for intensities
			if self.framepack_num == 3:
				self.shapes.colors = (rgb_value, rgb_value, rgb_value)
			else:
				self.shapes.colors = (rgb_value, 0, 0)
		else:
			self.shapes.colors = (0, 0, 0)
			

class non_gradient_ribbons(stim_protocol):
	"""
	Static "ribbons" in which the spatial gradient can be removed. Ribbons are 
	evenly spaced.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack: bool
		If True, pack 3 frames into single 16.67 ms flip
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	conv_cam_to_proj: conv_cam_to_proj method of convert_units class
		Converts units from camera pixels to stimulus units.
	ellipse_dx: floa
		Horizontal size of circle in stimulus units.
	ellipse_dy: float
		Vertical size of circle in stimulus units.
	width: float
		width of ribbon in mm.
	width_stim_units: float
		width of ribbon in stimulus units.
	num_ribbons: int
		Number of ribbons to plot.
	rbn_ys: list
		y positions of ribbons in projector units.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, width=2, num_ribbons=3, circle_diam=10):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		width: float
			Width of ribbon in mm
		num_ribbons: odd int
			Number of ribbons to plot.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		self.conv_cam_to_proj = convert_units().conv_cam_to_proj
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		stim_dx_per_px = (self.conv_cam_to_proj(1, 0)[0] - 
							self.conv_cam_to_proj(0, 0)[0])
		stim_dy_per_px = (self.conv_cam_to_proj(0, 1)[1] - 
							self.conv_cam_to_proj(0, 0)[1])
		self.ellipse_dx = circle_diam/mm_per_px*stim_dx_per_px
		self.ellipse_dy = circle_diam/mm_per_px*stim_dy_per_px
		self.width = width
		self.width_stim_units = width/mm_per_px*stim_dy_per_px
		
		# Set ribbon locations
		assert num_ribbons % 2 == 1, 'num_ribbons must be odd'
		self.num_ribbons = num_ribbons
		if self.num_ribbons > 1:
			self.rbn_ys = np.linspace(-0.85 + self.width_stim_units, 
									  0, num_ribbons//2 + 1)
			self.rbn_ys = np.hstack((self.rbn_ys[:-1], -self.rbn_ys[::-1]))
		else:
			self.rbn_ys = [0]
		self.rbn_ys = np.array(self.rbn_ys)
		
	def initialize_shapes(self):
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.max_flies, elementMask='circle',
						xys=np.zeros((self.max_flies, 2)), 
						sizes=[self.ellipse_dx, self.ellipse_dy],
						colors=self.shapes_color, elementTex=None)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Determine which flies are in the ribbon and illuminate them only.
		"""
		
		xs, ys = self.conv_cam_to_proj(centroids[:, 0], centroids[:, 1])
		
		# Distance of each fly to the closest ribbon
		min_dist_to_rbn = np.amin(abs(ys - self.rbn_ys[:, np.newaxis]), axis=0)
		
		# Find flies in any ribbon
		flies_in_rbn = 1*(min_dist_to_rbn < self.width_stim_units/2)
		
		# The 3*(1 - states)*3 term just pushes the boxes off the screen,
		# without having to change their color, which is time-consuming.
		# The 3*(1 - flies_in_rbn) pushes boxes off screen for flies not in
		# the ribbons.
		xys = [xs + 3*(1 - states) + 3*(1 - flies_in_rbn), ys]
		
		# Transpose list to (Nx2); faster than vstacking as numpy array
		vals = list(map(list, zip(*xys)))
		self.shapes.xys = vals

		
class tilted_static_ribbon(stim_protocol):
	"""
	Single static ribbon at arbitrary angle to wind flow.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack: bool
		If True, pack 3 frames into single 16.67 ms flip
	width: float
		width of ribbon in mm.
	width_stim_units: float
		width of ribbon in stimulus units.
	loc: float
		Placement of ribbon in % of projector coordinates.
	angle: float
			Angle relative to wind in degrees; 0 is parallel to flow.
	angle_stim_units: float
		Angle accounting for fact that projector is not square
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, width=2, loc=0, angle=0):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		width: float
			Width of ribbon in mm
		loc: float
			Y-placement of ribbon in % of projector coordinates
		angle: float
			Angle relative to wind in degrees; 0 is parallel to flow.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		conv_cam_to_proj = convert_units().conv_cam_to_proj
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		stim_dx_per_px = (conv_cam_to_proj(1, 0)[0] - 
						  conv_cam_to_proj(0, 0)[0])
		stim_dy_per_px = (conv_cam_to_proj(0, 1)[1] - 
						  conv_cam_to_proj(0, 0)[1])
		
		# This is only approximate because the transform in x and y are 
		# different. So the width of the ribbon may be off, even
		# subsantially.
		self.width = width
		self.width_stim_units = width/mm_per_px*stim_dy_per_px
		self.loc = loc
		
		dx = np.cos(angle*np.pi/180)
		dy = np.sin(angle*np.pi/180)
		self.angle = angle
		self.angle_stim_units = np.arctan(dy/dx*stim_dy_per_px/stim_dx_per_px)
		self.angle_stim_units *= 180/np.pi
	
	def initialize_shapes(self):	
		
		from psychopy import visual
		
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=1, fieldShape='square',
						elementMask=None, xys=np.array([[0, self.loc]]),
						sizes=[4, self.width_stim_units], 
						colors=self.shapes_color, 
						oris=[self.angle_stim_units],
						elementTex=None)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Nothing to update since static.
		"""
		
		pass


class whf_freq_by_loc(stim_protocol):
	"""
	Red circle illuminates the fly stochastically, but the frequency 
	increases depending on spatial location.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack_num: int
		If 3, packs RGB frames into single 16.67 ms flip; otherwise 1.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	conv_cam_to_proj: conv_cam_to_proj method of convert_units class
		Converts units from camera pixels to stimulus units.
	ellipse_dx: float
		Horizontal size of circle in stimulus units.
	ellipse_dy: float
		Vertical size of circle in stimulus units.
	x_power: positive int
		Power-dependence of profile of whiff frequency in x-direction
	plume_angle: float
		Angle of opening of plume, in degrees.
	plume_edge_fact: float
		Multiplier of x-position to get y-position of edge of cone, 
		in stimulus units.
	max_whf_freq: int
		Maximum whiff frequency in Hz
	whf_dur: float
		Whiff duration in milliseconds
	whf_frms: int
		Whiff duration in frames
	blnk_frms: int
		When a new whiff interrupts an old one, this separates them by 
		this many frames (allows high frequencies if whiffs are long)
	whf_count: array of shape (max_flies, )
		Holds the number of frames since beginning of whiff, for each fly.
		If zero, fly is not experiencing currently a whiff. Flies could 
		be "experiencing whiffs" if they are not active (state = 0); however
		they will not be illuminated if their state is 0.
	plume_flip: 1 or -1
		Flip plume across y-axis; if so, =-1
	moving_whfs: bool
		Whether whiffs are moving or static
	whf_dir_angle: float
		Angle of whiff motion in degrees
	whf_orientation: float
		Angle of whiff motion in degrees accounting for projector skew
	whf_px_x_per_frm, whf_px_y_per_frm: floats
		Number of stimulus [-1, 1] units the whiff moves each frame in x-, y-
	whf_width, whf_length: float
		Width of width in mm
	whf_width_px, whf_length_px: floats
		In stimulus units, the width and length of whiffs
	start_frms: int
		Number of frames before to start moving whiff; this is muliplied by
		whf_px_x_per_frm, whf_px_y_per_frm to get starting position of whiff
		behind fly center
	whf_xs, whf_ys: lists of length max_flies
		Location of whiffs for each fly.
	"""
		
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, circle_diam=10, x_power=1, 
				 plume_angle=30, max_whf_freq=5, whf_dur=150, 
				 blnk_dur=50, plume_flip=False, moving_whfs=False, whf_spd=150, 
				 whf_dir_angle=0, whf_width=3, whf_length=8, start_dist=15):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		circle_diam: float
			Diameter of stimulus circle covering each fly, in mm.
		x_power: positive int
			Power-dependence of profile of whiff frequency in x-direction
		plume_angle: int
			Angle of opening of plume, in degrees.
		max_whf_freq: int
			Maximum whiff frequency in Hz
		whf_dur: float
			Whiff duration in milliseconds
		blnk_dur: float
			When a new whiff interrupts an old one, this separates them by 
			this many milliseconds (allows high frequencies if whiffs are long)
		plume_flip: bool
			Flip plume across y-axis
		moving_whfs: bool
			Whether whiffs are moving or static
		whf_spd: float	
			Speed of whiffs in mm/s
		whf_dir_angle: float
			Angle of whiff motion in degrees
		whf_width, whf_length: floats
			x- and y-length of whiff in mm
		start_dist: float
			distance behind center of fly to start whiff motion in mm.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		self.config = load_settings()
		self.conv_cam_to_proj = convert_units().conv_cam_to_proj
		mm_per_px = self.config.getfloat("Camera", "mm_per_px")
		stim_dx_per_px = abs(self.conv_cam_to_proj(1, 0)[0] - 
							self.conv_cam_to_proj(0, 0)[0])
		stim_dy_per_px = abs(self.conv_cam_to_proj(0, 1)[1] - 
							self.conv_cam_to_proj(0, 0)[1])
		
		self.ellipse_dx = circle_diam/mm_per_px*stim_dx_per_px
		self.ellipse_dy = circle_diam/mm_per_px*stim_dy_per_px
		
		dx = np.cos(plume_angle*np.pi/180)
		dy = np.sin(plume_angle*np.pi/180)
		self.plume_angle = plume_angle
		self.plume_edge_fact = dy/dx*stim_dy_per_px/stim_dx_per_px
		self.x_power = x_power
		self.max_whf_freq = max_whf_freq
		self.whf_dur = whf_dur
		self.whf_frms = int(whf_dur/1000*self.rec_rate)
		self.blnk_frms = int(blnk_dur/1000*self.rec_rate)
		self.whf_count = np.zeros(self.max_flies)
		self.framepack_num = int(2*framepack + 1)
		
		# Reflect plume across y = 0?
		## TODO This may be incorrect: check
		self.plume_flip = 2*plume_flip - 1
			
		# Moving stimulus speed and angle; 0 deg means L to R
		self.moving_whfs = moving_whfs
		if self.moving_whfs == True:
			dx = (-1)**plume_flip*np.cos(whf_dir_angle*np.pi/180)
			dy = np.sin(whf_dir_angle*np.pi/180)
			self.whf_dir_angle = whf_dir_angle
			whf_dir_angle_stim_units = \
				np.arctan2(dy*stim_dy_per_px, dx*stim_dx_per_px)
			
			# This is in stimulus units (-1 to 1) per frame
			whf_spd_px_per_frm = abs(whf_spd/mm_per_px*stim_dx_per_px
									 /self.rec_rate/self.framepack_num)
			self.whf_px_x_per_frm = whf_spd_px_per_frm*\
				np.cos(whf_dir_angle_stim_units)
			self.whf_px_y_per_frm = whf_spd_px_per_frm*\
				np.sin(whf_dir_angle_stim_units)
			
			# Shape rotation in psychopy is defined reverse the usual way
			whf_dir_angle_stim_units *= 180/np.pi
			self.whf_orientation = -whf_dir_angle_stim_units
			
			# NOTE: this only works if angle is 0 degrees, otherwise 
			# differences in stim_dx_per_px and stim_dy_per_px can 
			# distort this. 
			self.whf_width = whf_width
			self.whf_length = whf_length
			self.whf_width_px = whf_width/mm_per_px*stim_dx_per_px
			self.whf_length_px = whf_length/mm_per_px*stim_dy_per_px
		
			# How far in negative time to start the whiff (so it will 
			# pass through fly rather than starting on it)
			self.start_frms = start_dist/mm_per_px*stim_dx_per_px/\
								whf_spd_px_per_frm
			
		self.whf_xs = np.zeros(self.max_flies)
		self.whf_ys = np.zeros(self.max_flies)
		
		"""
		# Check conversion is correct
		self.conv_proj_to_cam = convert_units().conv_proj_to_cam
		dx = np.cos(self.whf_dir_angle_stim_units*np.pi/180)
		dy = np.sin(self.whf_dir_angle_stim_units*np.pi/180)
		dx_cam = self.conv_proj_to_cam(dx, 0)[0] - self.conv_proj_to_cam(0, 0)[0]
		dy_cam = self.conv_proj_to_cam(0, dy)[1] - self.conv_proj_to_cam(0, 0)[1]
		print(dx_cam, dy_cam)
		"""
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		each fly. Each box is red; just moved off screen if fly disappears.
		"""
		
		from psychopy import visual
		if self.moving_whfs == False:
			self.shapes = visual.ElementArrayStim(self.win, 
							nElements=self.max_flies, elementMask='circle',
							xys=np.zeros((self.max_flies, 2)), 
							sizes=[self.ellipse_dx, self.ellipse_dy],
							colors=self.shapes_color, elementTex=None)
		else:
			self.shapes = visual.ElementArrayStim(self.win, 
							nElements=self.max_flies, 
							fieldShape='square', elementMask=None,
							xys=np.zeros((self.max_flies, 2)), 
							oris=self.whf_orientation,
							sizes=[self.whf_width_px, self.whf_length_px],
							colors=self.shapes_color, elementTex=None)			
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update whiff lengths. If new whiff, restart the whiff. If whiff 
		is more than whf_frms, end it.
		"""
		
		xs, ys = self.conv_cam_to_proj(centroids[:, 0], centroids[:, 1])
		
		"""
		# Visualize some fly positions to check that they are placed well
		from psychopy import visual
		xys_test = [xs + 3*(1 - states), ys]
		vals_test = list(map(list, zip(*xys_test)))
		self.shapes2 = []
		for i in range(20):
			self.shapes2.append(visual.Circle(self.win, radius=0.01, 
				lineColor='green', pos=vals_test[i]))	
			self.shapes2[i].draw()
		"""
		
		# Edge of plume y-position for given x-positions. Flies outside this 
		# y will not receive a whiff. 
		plume_edge_ys = self.plume_edge_fact*(1 - self.plume_flip*xs)/2.
		
		# X-rate depends on some power of x-locations 
		# Y-rate linearly drops away from plume until zero at plume edge
		x_rate_factor = (0.5*(1 + self.plume_flip*xs))**self.x_power
		y_rate_factor = np.maximum((abs(plume_edge_ys) - abs(ys))
						  /abs(plume_edge_ys), 0)
		rates = np.zeros(self.max_flies)
		rates = self.max_whf_freq*x_rate_factor*y_rate_factor
		
		# Whiffs occur with Poisson likelihood at rate `rates'
		whf_prob = 1 - np.exp(-rates/self.rec_rate/self.framepack_num)
		samples = np.random.uniform(0, 1, self.max_flies)		
		
		# Extend all existing whiffs an extra frame
		self.whf_count[self.whf_count > 0] += 1
		
		# Reset whiff frame to beginning of whiff if new hit
		self.whf_count[whf_prob > samples] = 1
		
		# If whiff is too long; end it
		self.whf_count[self.whf_count > (self.whf_frms + self.blnk_frms)] = 0
		
		# Non-active flies need whiff times of 0
		self.whf_count[states == 0] = 0
		
		# Any whiff separated by at least blnk_frms from last will be shown
		odor_on = self.whf_count > self.blnk_frms
		
		if self.moving_whfs == True:
		
			# Flies with whiffs: whiffs start at fly position minus some 
			# distance in negative time; this allows moving whiffs to properly 
			# cross fly body.
			self.whf_xs[self.whf_count == 0] = xs[self.whf_count == 0] 
			self.whf_ys[self.whf_count == 0] = ys[self.whf_count == 0] 
			self.whf_xs[self.whf_count == 1] = xs[self.whf_count == 1] - \
			  abs(self.start_frms)*self.whf_px_x_per_frm
			self.whf_ys[self.whf_count == 1] = ys[self.whf_count == 1] - \
			  abs(self.start_frms)*self.whf_px_y_per_frm
			self.whf_xs[self.whf_count > 1] += self.whf_px_x_per_frm
			self.whf_ys[self.whf_count > 1] += self.whf_px_y_per_frm
		else:
			self.whf_xs = xs
			self.whf_ys = ys
			
		# The 3.*(1. - states)*3 term just pushes the boxes off the screen,
		# without having to change their color, which is time-consuming. 
		# Also push off the boxes for flies which are not in a whiff event.
		#xys = [xs + 3*(1 - states) + 3*(1 - hit), ys]
		xys = [self.whf_xs + 3*(1 - states) + 3*(1 - odor_on), self.whf_ys]
		
		# Transpose list to (Nx2); faster than vstacking as numpy array
		vals = list(map(list, zip(*xys)))
		self.shapes.xys = vals


class movie(stim_protocol):
	"""
	Play a recorded movie throughout the video.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack_num: int
		If 3, packs RGB frames into single 16.67 ms flip; otherwise 1.
	stim_name: str;
		Name of stimulus video (in get_stim_vids_dir() dir), without extension
	stim_file: str;
		Name of stimulus video including extension
	size: 2-element array
		Size in pixels of frames
	loop, flip_horiz, flip_vert: bools
		Whether to loop or flip movie
	offset_pix: 2-element list
		Number of pixels to offset image in x and y.
	frame_Ts: list
		Timestamps in stimulus video, for each frame presented during 
		experiment.
	opacity: float between 0 and 1
		intensity of stimulus from transparent (0) to opaque (1)
	check_res: bool;
		If True, checks that resolution of movie and projector are same
	"""
	
	def __init__(self, win, rec_rate, max_flies, stim_name, invert=False,
				 framepack=False, flip_horiz=False, 
				 flip_vert=False, loop=False, offset_pix=[0, 0], intensity=1,
				 check_res=True, codec='mov'):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		stim_name: str;
			Name of stimulus (in get_stim_vids_dir() dir), without extension
		loop, flip_horiz, flip_vert: bools
			Whether to loop or flip movie
		offset_pix: 2-element list
			Number of pixels to offset image in x and y.
		intensity: float between 0 and 1
			alpha of stimulus from opaque (1) to transparent (0)
		check_res: bool;
			If True, checks that resolution of movie and projector are same
		codec: str
			movie type (avi, mp4, etc.)
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		self.stim_name = stim_name
		self.stim_file = stim_name + '.%s' % codec
		metadata = load_stim_video_metadata(stim_name)
		
		# Video playing attributes
		self.loop = loop
		self.flip_horiz = flip_horiz
		self.flip_vert = flip_vert
		self.offset_pix = offset_pix
		self.frame_Ts = []
		self.framepack_num = int(2*framepack + 1)
		self.size = metadata['size']
		self.opacity = intensity
		proj_mm_per_px_x = load_settings().getfloat("Projector", "mm_per_px_x")
		proj_mm_per_px_y = load_settings().getfloat("Projector", "mm_per_px_y")
		
		self.check_res = check_res
		if self.check_res == False:
			return
			
		assert metadata['proj_mm_per_px_x'] == proj_mm_per_px_x, \
			"mm per px for projector setup and for saved video are not equal;"\
			"this means distances are not faithful to real life"
		assert metadata['proj_mm_per_px_y'] == proj_mm_per_px_y, \
			"mm per px for projector setup and for saved video are not equal;"\
			"this means distances are not faithful to real life"
		assert metadata['fps'] == self.rec_rate*self.framepack_num, \
			"Frame rate of video (%s) must be equal to recording rate (%s) "\
			"times framepack number (%s)" % (metadata['fps'], self.rec_rate, 
			self.framepack_num)
		
	def initialize_shapes(self):	
		"""
		Constructor class for stimuli projection. 
		"""
		from psychopy import visual
		
		# These are just dummy shapes so the shape saving functions 
		# do not raise an Error. 
		self.shapes = visual.ElementArrayStim(self.win, 
							nElements=2, elementMask='circle',
							xys=3*np.ones((2, 2)), 
							sizes=[0.1, 0.1],
							colors=0, elementTex=None)
		
		filename=r'%s\%s' % (get_stim_vids_dir(), self.stim_file)

		self.screenshots = []
		self.vid_frames = visual.MovieStim2(self.win, filename=filename, 
				size=self.size, pos=self.offset_pix, loop=self.loop,
				flipVert=self.flip_vert, flipHoriz=self.flip_horiz, 
				opacity=self.opacity, noAudio=True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Only begin drawing video once the initialization time has passed.
		Save timestamp so we can pair to actualy play time of the video.
		"""
		
		if frm_num > 0:
			self.frame_Ts.append(self.vid_frames.getCurrentFrameTime())
			self.vid_frames.draw()
		

class scintillator(stim_protocol):
	"""
	Play a correlated noise stimulus with motion for some time, and 
	also include higher-order correlations as well as no motion during 
	off times.
	
	Each bar is `px_per_bar' pixels wide, and each bar is correlated with 
	the neighboring bar `t_step' frames later. The apparent speed of the 
	correlated stimulus is calculated as px_per_bar*mm_per_px*rec_rate,
	with an extra factor of 3 if framepacked. The mm_per_px depends on the 
	projector distance from the assay and is saved in settings.ini. Note 
	that framepacking advances the frames 3x as fast, which means that the 
	apparent speed can change. This is because the minimum spatial resolution 
	is set by the mm_per_px of the projector, so the maximum speed without
	framepacking is only mm_per_px*rec_rate. Framepacking increases this by
	3x.
	

	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	direction: int 1 or -1
		Whether motion is progressive (+x or +y) or regressive
	corr_sign: int 1 or -1
		Value of correlation: + or -
	px_per_bar: positive int
		Number of pixels for each bar
	t_step: positive int
		Number of frames to repeat each pattern. This slows down 
		apparent motion by this factor from the base rate.
	corr_x_step: positive int
		Pixels `corr_x_step' away from present pixel are correlated 
		at timestep `t_step'. When set to 1, neighboring pixels are 
		correlated over  t_step frames later. Apparent motion is 
		sped up by this factor. When corr_x_step and t_step are increased
		by the same factor, the motion has the same speed, but 
		correlated over a larger distance.
	switch_T: positive float
		Length of each static/motion cycle
	motion_T: positive float, less than switch T
		Length of the motion portion of the cycle
	num_cycle_frms: int
		Number of frames per cycle
	num_motion_frms: int
		Number of frames for scintillator
	cycle_frms: list of length of the total recording frames
		Contains the frame number within the cycle
	num_bars: int
		Number of independent flashing regions, presented as bars 
		perpendicular to the direction of motion. Each bar can be dark
		or illuminated; the pattern of these gives correlated motion.
	bar_size: int
		Size of each bar in stimulus units (-1 to 1)
	xs, ys: numpy array
		Locations of center of each bar, in stimulus units
	xys: numpy array with size (N, 2)
		Holds [xs, ys]
	C_motion: numpy array with size (N_T, num_bars)
		N_T is the number of frames for the cycle. This array holds the 
		bar illuminations (0 or 1 for Off/On), for each timestep in the cycle.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, motion_ori='x', 
				 direction=1, corr_sign=1, px_per_bar=1, t_step=1, 
				 corr_x_step=1, switch_T=3, motion_T=0.5):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		motion_ori: str 'x' or 'y'
			Direction of motion of the patterns
		direction: int 1 or -1
			Whether motion is progressive (+x or +y) or regressive
		corr_sign: int 1 or -1
			Value of correlation: + or -
		px_per_bar: positive int
			Number of pixels for each bar
		t_step: positive int
			Number of frames to repeat each pattern. This slows down 
			apparent motion by this factor from the base rate.
		corr_x_step: positive int
			Pixels `corr_x_step' away from present pixel are correlated 
			at timestep `t_step'. When set to 1, `neighboring' pixels are 
			correlated over  t_step frames later. Thus, apparent motion is 
			sped up by this factor. When corr_x_step and t_step are increased
			by the same factor, the motion has the same speed, but 
			correlations are spaced out spatially.
		switch_T: positive float
			Length of each static/motion cycle
		motion_T: positive float, less than switch T
			Length of the motion portion of the cycle
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		assert abs(direction) == 1, "direction must be +/-1"
		assert abs(corr_sign) == 1, "corr_sign must be +/-1"
		assert motion_T <= switch_T, "motion_T must be less/equal to switch_T"
		self.framepack_num = int(2*framepack + 1)
		
		self.motion_ori = motion_ori
		self.direction = direction
		self.corr_sign = corr_sign
		self.px_per_bar = px_per_bar
		self.t_step = t_step
		self.corr_x_step = corr_x_step
		self.switch_T = switch_T
		self.motion_T = motion_T
		self.num_cycle_frms = int(switch_T*self.rec_rate*self.framepack_num)
		self.num_motion_frms = int(motion_T*self.rec_rate*self.framepack_num)
		self.cycle_frms = []
		
		if motion_ori == 'y':
			self.num_bars = load_settings().getint("Projector", "res_dy")
			self.num_bars = self.num_bars//px_per_bar
			self.bar_size = [2, 2./self.num_bars]
			self.xs = np.zeros(self.num_bars)
			self.ys = np.linspace(-1, 1, self.num_bars)
			self.xys = np.vstack((self.xs, self.ys)).T
		elif motion_ori == 'x':
			self.num_bars = load_settings().getint("Projector", "res_dx")
			self.num_bars = self.num_bars//px_per_bar
			self.bar_size = [2./self.num_bars, 2]
			self.ys = np.zeros(self.num_bars)
			self.xs = np.linspace(-1, 1, self.num_bars)
			self.xys = np.vstack((self.xs, self.ys)).T
		else:
			print("motion_ori string %s must be `x` or `y`" % motion_ori)
			quit()
		
	def initialize_shapes(self):	
		"""
		Constructor class for stimuli projection. 
		"""
		from psychopy import visual
		
		# Each pixel in one direction gets a rectangle. These will be shifted
		# off and on the screen to show them.
		self.shapes = visual.ElementArrayStim(self.win, 
							nElements=self.num_bars, fieldShape='square',
							xys=self.xys, elementMask=None, 
							sizes=self.bar_size, 
							colors=self.shapes_color, 
							elementTex=None)
		self.shapes.setAutoDraw(True)
		
		# Moving stimulus defined by corr_sign, direction, t_step, num_corrs,
		# and corr_x_step. Motion speed is corr_x_step/t_step pixels per frame.
		# This gives mm_per_px*corr_x_step/t_step*framepack_num*rec_rate
		eta = np.random.normal(0, 1, (self.num_cycle_frms, self.num_bars))
		eta_shift = np.roll(np.roll(eta, 1, axis=0), self.direction, axis=1)
		self.C_motion = np.sign(eta + self.corr_sign*eta_shift
								*self.corr_x_step)
		self.C_motion = np.repeat(self.C_motion, self.t_step, axis=0)
		
		# Static with no correlation in +t
		eta = np.random.normal(0, 1, (self.num_cycle_frms, self.num_bars))
		self.C_static = np.sign(eta)
		self.C_static = np.repeat(self.C_static, self.t_step, axis=0)
		
		# Check that stimulus has correct correlation structure
		#T_shifts = range(-5, 5)
		#X_shifts = range(-5, 5)
		#vals = np.zeros((len(T_shifts), len(X_shifts)))
		#for idx, iT in enumerate(T_shifts):
		#	print(idx)
		#	for idy, jX in enumerate(X_shifts):
		#		C_1 = np.roll(np.roll(self.C_motion, iT, axis=0), jX, axis=1)
		#		vals[idx, idy] = np.mean(self.C_motion*C_1)
		#print(vals.flatten())
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Alternates motion and static at a frequency given by cycle_T
		"""
		
		# Repeats the motion stimulus every cycle_T seconds
		cycle_frm = np.mod(frm_num, self.num_cycle_frms)
		
		# Only for first motion_T seconds of cycle, the stimulus moves
		if cycle_frm < self.num_motion_frms:
			states = self.C_motion[cycle_frm]
		else:
			states = self.C_static[cycle_frm]
		
		# Zero contrast shapes are shifted off screen to show black.
		if self.motion_ori == 'y':
			self.xs = np.zeros(self.num_bars)
			self.xs[states == -1] = 3
		elif self.motion_ori == 'x':
			self.ys = np.zeros(self.num_bars)
			self.ys[states == -1] = 3
		self.shapes.xys = np.vstack((self.xs, self.ys)).T
		
		if frm_num > 0:
			self.cycle_frms.append(cycle_frm)


class scintillator_2(stim_protocol):
	"""
	Play a correlated noise stimulus with motion for some time, and 
	also include higher-order correlations as well as no motion during 
	off times.
	
	Each bar is `px_per_bar' pixels wide, and each bar is correlated with 
	the neighboring bar `t_step' frames later. The apparent speed of the 
	correlated stimulus is calculated as 
	px_per_bar*mm_per_px*rec_rate*corr_x_step/t_step,
	with an extra factor of 3 if framepacked. The mm_per_px depends on the 
	projector distance from the assay and is saved in settings.ini. Note 
	that framepacking advances the frames 3x as fast, which means that the 
	apparent speed can change. 
	
	This class generates stimuli that are correlated not only with 2
	neighboring pixels, but also  with i={1,...,`num_corrs'} nearest 
	neighbor pixels at time i*dt with ~exponentially decaying strength.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	direction: int 1 or -1
		Whether motion is progressive (+x or +y) or regressive
	corr_sign: int 1 or -1
		Value of correlation: + or -
	px_per_bar: positive int
		Number of pixels for each bar
	t_step: positive int
		Number of frames to repeat each pattern. This slows down 
		apparent motion by this factor from the base rate.
	corr_x_step: positive int
		Pixels `corr_x_step' away from present pixel are correlated 
		at timestep `t_step'. When set to 1, neighboring pixels are 
		correlated over  t_step frames later. Apparent motion is 
		sped up by this factor. When corr_x_step and t_step are increased
		by the same factor, the motion has the same speed, but 
		correlated over a larger distance.
	switch_T: positive float
		Length of each static/motion cycle
	motion_T: positive float, less than switch T
		Length of the motion portion of the cycle
	cycle_frms: list of length of the total recording frames
		Contains the frame number within the cycle
	num_corrs: int
		Number of offset correlations in time and space at (+/- i, i)
	static_motion_off: bool
		If True, then blank screen during off part of cycle
	seed: int
		Random number seed
	cycle_frms: list
		Holds the frame indicating point in the cycle 
	num_bars: int
		Number of independent flashing regions, presented as bars 
		perpendicular to the direction of motion. Each bar can be dark
		or illuminated; the pattern of these gives correlated motion.
	bar_size: int
		Size of each bar in stimulus units (-1 to 1)
	xs, ys: numpy array
		Locations of center of each bar, in stimulus units
	xys: numpy array size (N, 2)
		Holds [xs, ys]
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, motion_ori='x', 
				 direction=1, corr_sign=1, px_per_bar=1, t_step=1, 
				 corr_x_step=1, switch_T=3, motion_T=0.5, num_corrs=1,
				 static_motion_off=False, seed=0):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		motion_ori: str 'x' or 'y'
			Direction of motion of the patterns
		direction: int 1 or -1
			Whether motion is progressive (+x or +y) or regressive
		corr_sign: int 1 or -1
			Value of correlation: + or -
		px_per_bar: positive int
			Number of pixels for each bar
		t_step: positive int
			Number of frames to repeat each pattern. This slows down 
			apparent motion by this factor from the base rate.
		corr_x_step: positive int
			Pixels `corr_x_step' away from present pixel are correlated 
			at timestep `t_step'. When set to 1, `neighboring' pixels are 
			correlated over  t_step frames later. Thus, apparent motion is 
			sped up by this factor. When corr_x_step and t_step are increased
			by the same factor, the motion has the same speed, but 
			correlations are spaced out spatially.
		switch_T: positive float
			Length of each static/motion cycle
		motion_T: positive float, less than switch T
			Length of the motion portion of the cycle
		num_corrs: int
			Number of offset correlations in time and space at (+/- i, i)
		static_motion_off: bool
			If True, then blank screen during off part of cycle
		seed: int
			Random number seed
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		assert abs(direction) == 1, "direction must be +/-1"
		assert abs(corr_sign) == 1, "corr_sign must be +/-1"
		assert motion_T <= switch_T, "motion_T must be less/equal to switch_T"
		
		self.framepack_num = int(2*framepack + 1)
		self.motion_ori = motion_ori
		self.direction = direction
		self.corr_sign = corr_sign
		self.px_per_bar = px_per_bar
		self.t_step = t_step
		self.corr_x_step = corr_x_step
		self.switch_T = switch_T
		self.motion_T = motion_T
		self.num_cycle_frms = int(switch_T*self.rec_rate*self.framepack_num)
		self.num_motion_frms = int(motion_T*self.rec_rate*self.framepack_num)
		self.num_corrs = num_corrs
		self.static_motion_off = static_motion_off
		self.seed = seed
		self.cycle_frms = []
		
		if motion_ori == 'y':
			self.num_bars = load_settings().getint("Projector", "res_dy")
			self.num_bars = self.num_bars//px_per_bar
			self.bar_size = [2, 2./self.num_bars]
			self.ys = np.linspace(-1, 1, self.num_bars)
			self.xs = np.zeros(self.num_bars)
			self.xys = np.vstack((self.xs, self.ys)).T
			self.xys = self.xys
			
		elif motion_ori == 'x':
			self.num_bars = load_settings().getint("Projector", "res_dx")
			self.num_bars = self.num_bars//px_per_bar
			self.bar_size = [2./self.num_bars, 2]
			self.ys = np.zeros(self.num_bars)
			self.xs = np.linspace(-1, 1, self.num_bars)
			self.xys = np.vstack((self.xs, self.ys)).T
		else:
			print("motion_ori string %s must be `x` or `y`" % motion_ori)
			quit()
		
	def initialize_shapes(self):	
		"""
		Constructor class for stimuli projection. 
		"""
		from psychopy import visual
		
		# Each pixel in one direction gets a rectangle. These will be shifted
		# off and on the screen to show them.
		self.shapes = visual.ElementArrayStim(self.win, 
							nElements=self.num_bars, fieldShape='square',
							xys=self.xys, elementMask=None, 
							sizes=self.bar_size, 
							colors=self.shapes_color, 
							elementTex=None)
		self.shapes.setAutoDraw(True)
		
		# Moving stimulus defined by corr_sign, direction, t_step, num_corrs,
		# and corr_x_step. Motion speed is corr_x_step/t_step pixels per frame.
		# This gives mm_per_px*corr_x_step/t_step*framepack_num*rec_rate
		np.random.seed(self.seed)
		eta = np.random.normal(0, 1, (self.num_cycle_frms, self.num_bars))
		eta_shifts = np.zeros(eta.shape)
		for i in range(1, self.num_corrs + 1):
			eta_shift_i = np.roll(np.roll(eta, i, axis=0), 
								  self.direction*i*self.corr_x_step, axis=1)
			eta_shifts += self.corr_sign**i*eta_shift_i
		self.C_motion = np.sign(eta + eta_shifts)
		self.C_motion = np.repeat(self.C_motion, self.t_step, axis=0)
		
		# No correlations in x.
		eta = np.random.normal(0, 1, (self.num_cycle_frms, self.num_bars))
		self.C_static = np.sign(eta)
		self.C_static = np.repeat(self.C_static, self.t_step, axis=0)
		
		# If no stimulus at all during blanks
		if self.static_motion_off == True:
			self.C_static[:] = -1
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Alternates motion and static at a frequency given by cycle_T
		"""
		
		# Repeats the motion stimulus every cycle_T seconds
		cycle_frm = np.mod(frm_num, self.num_cycle_frms)
		
		# Only for first motion_T seconds of cycle, the stimulus moves
		if cycle_frm < self.num_motion_frms:
			states = self.C_motion[cycle_frm]
		else:
			states = self.C_static[cycle_frm]
		
		# Zero contrast shapes are shifted off screen to show black.
		if self.motion_ori == 'y':
			self.xs = np.zeros(self.num_bars)
			self.xs[states == -1] = 3
		elif self.motion_ori == 'x':
			self.ys = np.zeros(self.num_bars)
			self.ys[states == -1] = 3
		self.shapes.xys = np.vstack((self.xs, self.ys)).T
		
		if frm_num > 0:
			self.cycle_frms.append(cycle_frm)


class scintillator_3(stim_protocol):
	"""
	Play a correlated noise stimulus with motion for some time, and 
	also include higher-order correlations as well as no motion during 
	off times.
	
	Each bar is `px_per_bar' pixels wide, and each bar is correlated with 
	the neighboring bar `corr_t_step' frames later. Note that in contrast to 
	scintillator_2, `corr_t_step' refers to the time between correlated pixels,
	but time-adjacent pixels are NOT correlated -- i.e. while in 
	scintillator_2, `t_step=3' refers to a frame being repeated 3 times, 
	here it means that the the frame at t, t+dt, t+2dt, are uncorrelated,
	but t and t+3dt are correlated. To generate this, the Gaussian pattern
	is shifted in time by corr_t_step and in x by corr_x_step, then added to 
	the original pattern, and finally binary thresholded to get the 
	correlation structure. Note there is also the option to hold the pattern 
	for t frames, using `t_hold' -- this is analogous to `t_step' in 
	scintillator_2. Speed of the  correlated stimulus is calculated as 
	px_per_bar*mm_per_px*rec_rate/corr_t_step/t_hold*corr_x_step,
	with an extra factor of 3 if framepacked. The mm_per_px depends on the 
	projector distance from the assay and is saved in settings.ini. Note 
	that framepacking advances the frames 3x as fast, which means that the 
	apparent speed can change. 
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	direction: int 1 or -1
		Whether motion is progressive (+x or +y) or regressive
	corr_sign: int 1 or -1
		Value of correlation: + or -
	px_per_bar: positive int
		Number of pixels for each bar
	corr_x_step: positive int
		Pixels `corr_x_step' away from present pixel are correlated 
		at timestep `corr_t_step*t_hold'. When set to 1, neighboring pixels are 
		correlated over  corr_t_step*t_hold frames later. Apparent motion is 
		sped up by this factor. 
	corr_t_step: positive int
		Number of frames separating correlation: <x,t ; x, t + corr_t_step> > 0
		This slows down apparent motion by this factor from the base rate.
	t_hold: positive int
		Number of frames to repeat before advancing stimulus.
		This slows down apparent motion by this factor from the base rate.
	switch_T: positive float
		Length of each static/motion cycle
	motion_T: positive float, less than switch T
		Length of the motion portion of the cycle
	cycle_frms: list of length of the total recording frames
		Contains the frame number within the cycle
	static_motion_off: bool
		If True, then blank screen during off part of cycle
	seed: int
		Random number seed
	cycle_frms: list
		Holds the frame indicating point in the cycle 
	num_bars: int
		Number of independent flashing regions, presented as bars 
		perpendicular to the direction of motion. Each bar can be dark
		or illuminated; the pattern of these gives correlated motion.
	bar_size: int
		Size of each bar in stimulus units (-1 to 1)
	xs, ys: numpy array
		Locations of center of each bar, in stimulus units
	xys: numpy array size (N, 2)
		Holds [xs, ys]
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, motion_ori='x', 
				 direction=1, corr_sign=1, px_per_bar=1, corr_t_step=1, 
				 t_hold=1, corr_x_step=1, switch_T=3, motion_T=0.5, 
				 static_motion_off=False, seed=0):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		motion_ori: str 'x' or 'y'
			Direction of motion of the patterns
		direction: int 1 or -1
			Whether motion is progressive (+x or +y) or regressive
		corr_sign: int 1 or -1
			Value of correlation: + or -
		px_per_bar: positive int
			Number of pixels for each bar
		corr_x_step: positive int
			Pixels `corr_x_step' away from present pixel are correlated 
			at timestep `corr_t_step'. When set to 1, `neighboring' pixels are 
			correlated over  corr_t_step frames later. Thus, apparent motion is 
			sped up by this factor. When corr_x_step and corr_t_step are increased
			by the same factor, the motion has the same speed, but 
			correlations are spaced out spatially.
		corr_t_step: positive int
			Number of frames separating correlation: <x,t ; x, t 
			+ corr_t_step> > 0. This slows down apparent motion by this factor 
			from the base rate.
		t_hold: positive int
			Number of frames to hold stimulus before advancing.
			This slows down apparent motion by this factor from the base rate.
		switch_T: positive float
			Length of each static/motion cycle
		motion_T: positive float, less than switch T
			Length of the motion portion of the cycle
		static_motion_off: bool
			If True, then blank screen during off part of cycle
		seed: int
			Random number seed
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		assert abs(direction) == 1, "direction must be +/-1"
		assert abs(corr_sign) == 1, "corr_sign must be +/-1"
		assert motion_T <= switch_T, "motion_T must be less/equal to switch_T"
		
		self.framepack_num = int(2*framepack + 1)
		self.motion_ori = motion_ori
		self.direction = direction
		self.corr_sign = corr_sign
		self.px_per_bar = px_per_bar
		self.corr_x_step = corr_x_step
		self.corr_t_step = corr_t_step
		self.t_hold = t_hold
		self.switch_T = switch_T
		self.motion_T = motion_T
		self.num_cycle_frms = int(switch_T*self.rec_rate*self.framepack_num)
		self.num_motion_frms = int(motion_T*self.rec_rate*self.framepack_num)
		self.static_motion_off = static_motion_off
		self.seed = seed
		self.cycle_frms = []
		
		if motion_ori == 'y':
			self.num_bars = load_settings().getint("Projector", "res_dy")
			self.num_bars = self.num_bars//px_per_bar
			self.bar_size = [2, 2./self.num_bars]
			self.ys = np.linspace(-1, 1, self.num_bars)
			self.xs = np.zeros(self.num_bars)
			self.xys = np.vstack((self.xs, self.ys)).T
			self.xys = self.xys
			
		elif motion_ori == 'x':
			self.num_bars = load_settings().getint("Projector", "res_dx")
			self.num_bars = self.num_bars//px_per_bar
			self.bar_size = [2./self.num_bars, 2]
			self.ys = np.zeros(self.num_bars)
			self.xs = np.linspace(-1, 1, self.num_bars)
			self.xys = np.vstack((self.xs, self.ys)).T
		else:
			print("motion_ori string %s must be `x` or `y`" % motion_ori)
			quit()
		
	def initialize_shapes(self):	
		"""
		Constructor class for stimuli projection. 
		"""
		from psychopy import visual
		
		# Each pixel in one direction gets a rectangle. These will be shifted
		# off and on the screen to show them.
		self.shapes = visual.ElementArrayStim(self.win, 
							nElements=self.num_bars, fieldShape='square',
							xys=self.xys, elementMask=None, 
							sizes=self.bar_size, 
							colors=self.shapes_color, 
							elementTex=None)
		self.shapes.setAutoDraw(True)
		
		# Moving stimulus defined by corr_sign, direction, corr_t_step, t_hold,
		# and corr_x_step. Motion speed is corr_x_step/corr_t_step/t_repeat 
		# pixels per frame. This gives mm_per_px*corr_x_step/corr_t_step*
		# framepack_num*rec_rate sa the speed
		np.random.seed(self.seed)
		eta = np.random.normal(0, 1, (self.num_cycle_frms, self.num_bars))
		eta_shift = self.corr_sign*np.roll(np.roll(eta, self.corr_t_step, 
					  axis=0), self.direction*self.corr_x_step, axis=1)
		self.C_motion = np.sign(eta + eta_shift)
		self.C_motion = np.repeat(self.C_motion, self.t_hold, axis=0)
		
		# No correlations in x.
		eta = np.random.normal(0, 1, (self.num_cycle_frms, self.num_bars))
		self.C_static = np.sign(eta)
		self.C_static = np.repeat(self.C_static, self.t_hold, axis=0)
		
		# If no stimulus at all during blanks
		if self.static_motion_off == True:
			self.C_static[:] = -1
		
		# Check that stimulus has correct correlation structure
		T_shifts = range(-5, 6)
		X_shifts = range(-5, 6)
		vals = np.zeros((len(T_shifts), len(X_shifts)))
		for idx, iT in enumerate(T_shifts):
			print(idx)
			for idy, jX in enumerate(X_shifts):
				C_1 = np.roll(np.roll(self.C_motion, iT, axis=0), jX, axis=1)
				vals[idx, idy] = np.mean(self.C_motion*C_1)
		print(np.around(vals, 2))
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Alternates motion and static at a frequency given by cycle_T
		"""
		
		# Repeats the motion stimulus every cycle_T seconds
		cycle_frm = np.mod(frm_num, self.num_cycle_frms)
		
		# Only for first motion_T seconds of cycle, the stimulus moves
		if cycle_frm < self.num_motion_frms:
			states = self.C_motion[cycle_frm]
		else:
			states = self.C_static[cycle_frm]
		
		# Zero contrast shapes are shifted off screen to show black.
		if self.motion_ori == 'y':
			self.xs = np.zeros(self.num_bars)
			self.xs[states == -1] = 3
		elif self.motion_ori == 'x':
			self.ys = np.zeros(self.num_bars)
			self.ys[states == -1] = 3
		self.shapes.xys = np.vstack((self.xs, self.ys)).T
		
		if frm_num > 0:
			self.cycle_frms.append(cycle_frm)


class noisy_glider(stim_protocol):
	"""
	Play a correlated noise stimulus with motion for some time. 
	The stimulus is a glider so contains +1 correlation for all +jx, +jt.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	direction: int 1 or -1
		Whether motion is progressive (+x or +y) or regressive
	corr_sign: int 1 or -1
		Value of correlation: + or -
	px_per_bar: positive int
		Number of pixels for each bar
	t_step: positive int
		Number of frames to repeat each pattern. This slows down 
		apparent motion by this factor from the base rate.
	corr_x_step: positive int
		Pixels `corr_x_step' away from present pixel are correlated 
		at timestep `t_step'. When set to 1, neighboring pixels are 
		correlated over  t_step frames later. Apparent motion is 
		sped up by this factor. When corr_x_step and t_step are increased
		by the same factor, the motion has the same speed, but 
		correlated over a larger distance.
	corr_x_step_blank: bool
		If True, reduces the effective spatial resolution by blanking
		all bars from 1 to corr_x_step. In other words, the signal jumps
		by corr_x_step at each frame, but all bars in between these
		never have signal. This reduces the effective resolution of the
		projector.
	switch_T: positive float
		Length of each static/motion cycle
	motion_T: positive float, less than switch T
		Length of the motion portion of the cycle
	cycle_frms: list of length of the total recording frames
		Contains the frame number within the cycle
	num_corrs: int
		Number of offset correlations in time and space at (+/- i, i)
	static_motion_off: bool
		If True, then blank screen during off part of cycle
	seed: int
		Random number seed
	cycle_frms: list
		Holds the frame indicating point in the cycle 
	num_bars: int
		Number of independent flashing regions, presented as bars 
		perpendicular to the direction of motion. Each bar can be dark
		or illuminated; the pattern of these gives correlated motion.
	bar_size: int
		Size of each bar in stimulus units (-1 to 1)
	xs, ys: numpy array
		Locations of center of each bar, in stimulus units
	xys: numpy array size (N, 2)
		Holds [xs, ys]
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, motion_ori='x', 
				 direction=1, corr_sign=1, px_per_bar=1, t_step=1, 
				 corr_x_step=1, corr_x_step_blank=False, switch_T=3, 
				 motion_T=0.5, static_motion_off=False, seed=0):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		motion_ori: str 'x' or 'y'
			Direction of motion of the patterns
		direction: int 1 or -1
			Whether motion is progressive (+x or +y) or regressive
		corr_sign: int 1 or -1
			Value of correlation: + or -
		px_per_bar: positive int
			Number of pixels for each bar
		t_step: positive int
			Number of frames to repeat each pattern. This slows down 
			apparent motion by this factor from the base rate.
		corr_x_step: positive int
			Pixels `corr_x_step' away from present pixel are correlated 
			at timestep `t_step'. When set to 1, neighboring pixels are 
			correlated over  t_step frames later. Apparent motion is 
			sped up by this factor. When corr_x_step and t_step are increased
			by the same factor, the motion has the same speed, but 
			correlated over a larger distance.
		corr_x_step_blank: bool
			If True, reduces the effective spatial resolution by blanking
			all bars from 1 to corr_x_step. In other words, the signal jumps
			by corr_x_step at each frame, but all bars in between these
			never have signal. This reduces the effective resolution of the
			projector.
		switch_T: positive float
			Length of each static/motion cycle
		motion_T: positive float, less than switch T
			Length of the motion portion of the cycle
		num_corrs: int
			Number of offset correlations in time and space at (+/- i, i)
		static_motion_off: bool
			If True, then blank screen during off part of cycle
		seed: int
			Random number seed
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		assert abs(direction) == 1, "direction must be +/-1"
		assert abs(corr_sign) == 1, "corr_sign must be +/-1"
		assert motion_T <= switch_T, "motion_T must be less/equal to switch_T"
		
		self.framepack_num = int(2*framepack + 1)
		self.motion_ori = motion_ori
		self.direction = direction
		self.corr_sign = corr_sign
		self.px_per_bar = px_per_bar
		self.t_step = t_step
		self.corr_x_step = corr_x_step
		self.corr_x_step_blank = corr_x_step_blank
		self.switch_T = switch_T
		self.motion_T = motion_T
		self.num_cycle_frms = int(switch_T*self.rec_rate*self.framepack_num)
		self.num_motion_frms = int(motion_T*self.rec_rate*self.framepack_num)
		self.static_motion_off = static_motion_off
		self.seed = seed
		self.cycle_frms = []
		
		if motion_ori == 'y':
			self.num_bars = load_settings().getint("Projector", "res_dy")
			self.num_bars = self.num_bars//px_per_bar
			self.bar_size = [2, 2./self.num_bars]
			self.xs = np.zeros(self.num_bars)
			self.ys = np.linspace(-1, 1, self.num_bars)
			self.xys = np.vstack((self.xs, self.ys)).T
		elif motion_ori == 'x':
			self.num_bars = load_settings().getint("Projector", "res_dx")
			self.num_bars = self.num_bars//px_per_bar
			self.bar_size = [2./self.num_bars, 2]
			self.ys = np.zeros(self.num_bars)
			self.xs = np.linspace(-1, 1, self.num_bars)
			self.xys = np.vstack((self.xs, self.ys)).T
		else:
			print("motion_ori string %s must be `x` or `y`" % motion_ori)
			quit()
		
	def initialize_shapes(self):	
		"""
		Constructor class for stimuli projection. 
		"""
		from psychopy import visual
		
		# Each pixel in one direction gets a rectangle. These will be shifted
		# off and on the screen to show them.
		self.shapes = visual.ElementArrayStim(self.win, 
							nElements=self.num_bars, fieldShape='square',
							xys=self.xys, elementMask=None, 
							sizes=self.bar_size, 
							colors=self.shapes_color, 
							elementTex=None)
		self.shapes.setAutoDraw(True)
		
		# Moving stimulus defined by corr_sign, direction, t_step,
		# and corr_x_step. Motion speed is corr_x_step/t_step pixels per frame.
		# This gives mm_per_px*corr_x_step/t_step*framepack_num*rec_rate
		np.random.seed(self.seed)
		eta = np.random.normal(0, 1, (self.num_cycle_frms, self.num_bars))
		self.C_motion = np.sign(eta)
		for i in range(eta.shape[0]):
			self.C_motion[i, :] = np.roll(self.C_motion[0, :], 
										  self.direction*i*self.corr_x_step, 
										  axis=0)*self.corr_sign**i
		
		# Blacks out corr_x_step - 1 frames to effectively reduce the spatial 
		# resolution by interleaving with no signal
		if self.corr_x_step_blank == True:
			for j in range(1, self.corr_x_step):
				self.C_motion[:, j::self.corr_x_step] = -1
		self.C_motion = np.repeat(self.C_motion, self.t_step, axis=0)
		
		# No correlations in x.
		eta = np.random.normal(0, 1, (self.num_cycle_frms, self.num_bars))
		self.C_static = np.sign(eta)
		if self.corr_x_step_blank == True:
			for j in range(1, self.corr_x_step):
				self.C_static[:, j::self.corr_x_step] = -1
		self.C_static = np.repeat(self.C_static, self.t_step, axis=0)
		
		# If no stimulus at all during blanks
		if self.static_motion_off == True:
			self.C_static[:] = -1
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Alternates motion and static at a frequency given by cycle_T
		"""
		
		# Repeats the motion stimulus every cycle_T seconds
		cycle_frm = np.mod(frm_num, self.num_cycle_frms)
		
		# Only for first motion_T seconds of cycle, the stimulus moves
		if cycle_frm < self.num_motion_frms:
			states = self.C_motion[cycle_frm]
		else:
			states = self.C_static[cycle_frm]
		
		# Zero contrast shapes are shifted off screen to show black.
		if self.motion_ori == 'y':
			self.xs = np.zeros(self.num_bars)
			self.xs[states == -1] = 3
		elif self.motion_ori == 'x':
			self.ys = np.zeros(self.num_bars)
			self.ys[states == -1] = 3
		self.shapes.xys = np.vstack((self.xs, self.ys)).T
		
		if frm_num > 0:
			self.cycle_frms.append(cycle_frm)


class full_field_flash_telegraph(stim_protocol):
	"""
	Flash entire arena with a signal defined by a stochastic telegraph process.
	Flashes are defined by either switching rates s1 and s2, or by frequency
	and duration. If s1 and s2 are passed, avg_freq and avg_dur must be passed 
	as None, and vice versa. 
	Alternatively, a binary sequence can be input to define when flashes occur.
	In addition, flashes occur in `blocks", the length and period 
	of which are separately defined:
	
				  <------ON block------><---OFF block---><------ON block------>
	Odor signal: |-|-|_|-|__|-|-|-|__|-|_______________|-|-|_|-|__|-|-|-|__|-|
				 __    
				 Flash duration
	
	If `sequence' is passed, avg_dur, avg_freq, s1, and s2 are ignored
	
	Attrs
	--------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack: bool
		If True, pack 3 frames into single 16.67 ms flip
	avg_freq: float
		Mean frequency of flashes in frames/sec. This is the frequencey of 
		the flashes only; so it is 2x the telegraph switching rate.
	avg_dur: float
		Mean duration of flash in seconds. 
	On_T: float
		Length of block in seconds during which flashes are delivered
	Off_T: float
		Length of block in seconds during which no flashes are delivered. 
		This is cycled with On_T to give a total block period of On_T+Off_T
	num_On_Off_frms: int
		Number of frames between start of each stimulus block (ie period 
		of blocks)
	num_On_frms: int
		Number of frames for each ON part of the cycle (ie number of frames
		during which flashes will occur)
	avg_blank_dur: float
		Mean length of a blank in seconds
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	s1: float
		Switching rate OFF --> ON in Hz
	s2: float
		Switching rate ON --> OFF in Hz  
	intermittency: float 0 < 1
		Intermittency of signal = avg blank length / cycle length
	cycle_frames: int
		Number of frames in a block 
	seed: int
		Random number seed
	
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, s1=1, s2=1, avg_freq=None, avg_dur=None, 
				 sequence=None, On_T=15.0, Off_T=15.0, seed=0):
		
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid.
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		s1: float or None
			switching rate OFF --> ON in Hz
		s2: float or None
			switching rate ON --> OFF in Hz  
		avg_freq: float
			Mean frequency of flashes in frames/sec. This is the frequencey of 
			the flash onsets; so it is 2x the telegraph switching rate.
		avg_dur: float
			Mean duration of a flash in seconds. Flash durations are themselves 
			exponentially distributed; this is just the mean.
		sequence: list or None
			Array of 1s and 0s, where 1 indicates a flash and 0 indicates no 
			flash. Must be same length as On_T in frames. If passed, avg_freq, 
			avg_dur, s1, and s2 are ignored.
		On_T: float
			Length of block in seconds during which flashes are delivered
		Off_T: float
			Length of block in seconds during which no flashes are delivered. 
			This is cycled with On_T to give a total block period of On_T+Off_T
		"""
		super().__init__(win, rec_rate, max_flies, invert, framepack)
	
		self.framepack_num = int(2*framepack + 1)
		self.rec_rate = rec_rate
		
		# Sum of On and Off gives full cycle
		self.On_T = On_T
		self.Off_T = Off_T
		self.num_On_Off_frms = int((self.Off_T + self.On_T)*
								  self.rec_rate*self.framepack_num)
		self.num_On_frms = int(self.On_T*self.rec_rate*self.framepack_num)
		
		if sequence is not None:
			assert len(sequence) == int(On_T*rec_rate*self.framepack_num), \
				"Sequence must be the same length as On_T in frames; 3x"\
				"if framepacked"
			self.sequence = sequence
			return None
		
		if (s1 is None) and (s2 is None):
			assert ((avg_dur is not None) and (avg_freq is not None)), "Both "\
			  "avg_freq and avg_dur must be set to use avg_freq and avg_dur"
			assert avg_dur > 0, "avg_dur must be positive and defined in sec"
			assert avg_freq > 0, "avg_freq must be >0, defined in frames/sec"

			self.avg_freq = avg_freq
			self.avg_dur = avg_dur
			self.avg_blank_dur = 1/self.avg_freq - self.avg_dur
			assert self.avg_blank_dur > 0, "Decrease avg_dur or decrease avg_freq"

			self.s1 = 1/self.avg_blank_dur 
			self.s2 = 1/self.avg_dur

		elif (avg_freq is None) and (avg_dur is None):
			assert ((s1 is not None) and (s2 is not None)), "Both s1 and s2 "\
			  "must be set to use s1 and s2"
			assert s1 > 0, "s1 must be greater than 0"
			assert s2 > 0, "s2 must be greater than 0"        
			
			self.s1 = s1
			self.s2 = s2
			self.avg_dur = 1/self.s2
			self.avg_freq = (self.s1*self.s2)/(self.s1 + self.s2)
			self.avg_blank_dur = 1/self.avg_freq - self.avg_dur
			
		else:
			print("Must pass either avg_freq and avg_dur or s1 and s2. If "
			  "s1 and s2 passed, then avg_freq and avg_dur must be None, "
			  "and vice versa")
			quit()
		
		self.intermittency = self.s2/(self.s1 + self.s2) 
		self.cycle_frames = int(self.On_T*self.rec_rate*self.framepack_num)
		self.seed = seed

		np.random.seed(self.seed)        
		num_frames = 0
		self.sequence = []
		while num_frames < self.cycle_frames:
			blank_dur = np.random.exponential(scale=1/self.s1)
			whiff_dur = np.random.exponential(scale=1/self.s2)
				
			# Minimum length is 1 frame
			blank_frames = max(int(np.round(blank_dur*self.rec_rate*
							self.framepack_num, 0)), 1)
			whiff_frames = max(int(np.round(whiff_dur*self.rec_rate*
							self.framepack_num, 0)), 1)
			self.sequence.extend([0]*blank_frames)    
			self.sequence.extend([1]*whiff_frames)    
			num_frames += blank_frames + whiff_frames
			
		self.sequence = self.sequence[:self.cycle_frames]
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		whole screen.
		"""
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=1, elementMask=None,
						xys=[[0, 0]], fieldShape='sqr',
						sizes=[[2, 2]], colors=self.shapes_color, 
						elementTex=None)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update position if on frame number to update. Make black by moving box
		too far to right, off screen. 
		"""
		
		# Frame modded the On/Off block cycle
		block_frm = np.mod(frm_num, self.num_On_Off_frms)
		
		# If within the ON block, then go through the sequence
		if block_frm < self.num_On_frms:
			if self.sequence[block_frm] == 1:
				self.shapes.xys = [[0, 0]]
			else:
				self.shapes.xys = [[3, 0]]
		else:
			self.shapes.xys = [[3, 0]]


class lateral_bars_in_plume_cone(stim_protocol):
	"""
	Bars moving outward from the centerline within a plume cone region.
	
	Attrs
	-------

	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	stim_dx_per_px, stim_dx_per_px: floats
		Conversion factors from stimulus units (-1 to 1) to pixels. This 
		allows the plume angle to be defined correctly in degrees, accounting
		for the fact that stimulus units in x and y are not equal.
	on_width: float
		Width of bar in mm.
	off_width: float
		Width of region between bar in mm. If None, set to on_width
	on_width_stim_units: float
		Width of bar in stimulus units.
	off_width_stim_units: float
		Width of space between bars in stimulus units.
	num_bars: int
		Number of total bar shapes in psychopy object
	speed: float
		Speed in mm/s
	proj_speed: float
		Speed of bar in stimulus units per frame.
	invert: bool
		Bars are black; background is red
	framepack_num: int
		3 if packing RGB frames; 1 otherwise
	xs: List of length num_bars.
		x-positions of the bars
	ys: list of length num_bars.
		y-positions of the bars
	ys1: list of length num_bars/2.
		y-positions of the bars on one side of plume centerline
	ys2: list of length num_bars/2.
		y-positions of the bars on other side of plume centerline
	On_T, Off_T: floats
		If both are not None, then bars are presented in blocks of
		On_T seconds, interrupted by Off_T seconds of no stimulus.
	plume_angle: positive float between 0 and 90
			Angle of plume cone opening, where 0 degrees is no cone and 90
			is the full arena
	plume_flip: bool
		Flip plume across y-axis?
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, speed=15, on_width=5, off_width=None,
				 On_T=None, Off_T=None, plume_angle=30, plume_flip=False):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack_num: int
			3 if packing RGB frames; 1 otherwise
		speed: float
			Speed of bar in mm/s.
		on_width: float
			Width of bar in mm.
		off_width: float
			Width of region between bar in mm. If None, set to on_width
		invert: bool
			Bars are black; background is red
		On_T, Off_T: floats
			If both are not None, then bars are presented in blocks of
			On_T seconds, interrupted by Off_T seconds of no stimulus.
		plume_angle: positive float between 0 and 90
			Angle of plume cone opening, where 0 degrees is no cone and 90 
			is full arena
		plume_flip: bool
			Flip plume across y-axis?
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		conv_cam_to_proj = convert_units().conv_cam_to_proj
		self.stim_dx_per_px = abs(conv_cam_to_proj(1, 0)[0] - 
							conv_cam_to_proj(0, 0)[0])
		self.stim_dy_per_px = abs(conv_cam_to_proj(0, 1)[1] - 
							conv_cam_to_proj(0, 0)[1])
		
		if off_width is None:
			off_width = on_width
		self.on_width = on_width
		self.off_width = off_width
		self.on_width_stim_units = on_width/mm_per_px*self.stim_dy_per_px
		self.off_width_stim_units = off_width/mm_per_px*self.stim_dy_per_px
		
		self.speed = speed
		self.proj_speed = speed/self.rec_rate/mm_per_px*self.stim_dy_per_px
		
		self.plume_flip = plume_flip
		self.plume_angle = plume_angle
		
		self.framepack_num = int(2*framepack + 1)
		
		if (On_T is not None) and (Off_T is not None):
			self.On_T = On_T
			self.Off_T = Off_T
			self.num_On_Off_frms = int((Off_T + On_T)*\
									 rec_rate*self.framepack_num)
			self.num_On_frms = int(On_T*rec_rate*self.framepack_num)
		else:
			self.On_T = None
			self.Off_T = None

	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Bars evenly spaced on 
		screen from top to bottom.
		"""
		
		# Width in stimulus units of 1 cycle
		self.cycle_width = self.on_width_stim_units + self.off_width_stim_units
		
		# Min # of whiff + blank cycles needed to fully cover arena
		cycles_per_arena = 2//self.cycle_width + 1
		self.full_width = 3*cycles_per_arena*self.cycle_width
		
		# ys1 and ys2 are on different sides of plume centerline
		self.ys1 = np.arange(0, self.full_width, self.cycle_width)
		self.ys2 = np.arange(0, -self.full_width, -self.cycle_width)
		self.ys = np.hstack((self.ys1, self.ys2))
		self.num_bars = len(self.ys1) + len(self.ys2)
				
		# Bars move in x direction given by angle of plume
		self.stim_fact = self.stim_dx_per_px/self.stim_dy_per_px
		self.tan_angles = abs(np.tan(self.plume_angle*np.pi/180))
		self.xs = abs(self.ys)*self.stim_fact/self.tan_angles
		self.xs *= (-1)**self.plume_flip
		
		xys = [self.xs, self.ys]
		xys = list(map(list, zip(*xys)))
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.num_bars, elementMask=None,
						xys=xys, fieldShape='sqr',
						sizes=[2, self.on_width_stim_units],
						colors=self.shapes_color, elementTex=None)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update positions to move by one step; cycle at boundaries.
		"""
		
		self.ys1 += self.proj_speed/self.framepack_num
		self.ys2 -= self.proj_speed/self.framepack_num
		self.ys = np.hstack((self.ys1, self.ys2))
		
		# If using blocks, then push bars off screen during blank period
		dx = 0
		if (self.On_T is not None) and (self.Off_T is not None):
			block_frm = np.mod(frm_num, self.num_On_Off_frms)
			if (block_frm > self.num_On_frms):
				dx = 3
			
		self.tan_angles = abs(np.tan(self.plume_angle*np.pi/180))
		self.xs = abs(self.ys)*self.stim_fact/self.tan_angles + dx
		self.xs *= (-1)**self.plume_flip
		
		xys = [self.xs, self.ys]
		xys = list(map(list, zip(*xys)))
		self.ys1[self.ys1 > self.full_width] = 0
		self.ys1[self.ys1 < 0] = self.full_width
		self.ys2[self.ys2 < -self.full_width] = 0
		self.ys2[self.ys2 > 0] = -self.full_width
		self.shapes.xys = xys


# temporary; to run prelim data experiments
# class full_field_frequency_gradient(stim_protocol):
class frequency_gradient_individual_flies(stim_protocol):
	"""
	Position-based red circle flashes the fly given its position.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	conv_cam_to_proj: conv_cam_to_proj method of convert_units class
		Converts units from camera pixels to stimulus units.
	ellipse_dx: float
		Horizontal size of circle in stimulus units.
	ellipse_dy: float
		Vertical size of circle in stimulus units.
	"""
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, intensity=35, circle_diam=10,
				 gradient_dir="dx", gradient_space="linear",
				 num_bands=0, update_interval=0.3,
				 min_freq=1, max_freq=5, hit_duration=0,
				 On_T=15, Off_T=15):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		intensity: int
			The rgb255 intensity for the red light displayed on the projector.
		circle_diam: float
			Diameter of stimulus circle covering each fly, in mm.
		gradient_dir: str
			Direction of the gradient in the arena (±dx or ±dy).
		gradient_space: str
			Gradient scale along arena (linear or log space).
		num_bands: int
			Number of frequency bands (0 is continous).
		update_interval: float
			Interval to update frequency stimulus per fly, in seconds.
		min_freq: int
			Lowest frequency band stimulus, in Hz.
		max_freq: int
			Highest frequency band stimulus, in Hz.
		hit_duration: float
			duration for each odor hit, in seconds.
			if 0, use 50% duty cycle
		On_T: int
			Duration of stimuli presentation, in seconds.
		Off_T: int
			Duration of blank presentation, in seconds.
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		self.config = load_settings()
		self.conv_cam_to_proj = convert_units().conv_cam_to_proj
		mm_per_px = self.config.getfloat("Camera", "mm_per_px")
		stim_dx_per_px = (self.conv_cam_to_proj(1, 0)[0]
						  - self.conv_cam_to_proj(0, 0)[0])
		stim_dy_per_px = (self.conv_cam_to_proj(0, 1)[1]
						  - self.conv_cam_to_proj(0, 0)[1])
		self.ellipse_dx = circle_diam/mm_per_px*stim_dx_per_px
		self.ellipse_dy = circle_diam/mm_per_px*stim_dy_per_px

		self.framepack_num = int(2*framepack + 1)
		self.intensity = intensity
		self.grad_dir = gradient_dir
		self.gradient_space = gradient_space
		self.num_bands = num_bands
		self.min_freq = min_freq if min_freq > 0 else 0.1
		self.max_freq = max_freq
		self.hit_duration = hit_duration
		self.update_interval = update_interval

		assert (self.grad_dir in ["dx", "dy", "-dx", "-dy", ]), \
		  f"gradient direction '{self.grad_dir}' not supported."

		assert (self.rec_rate*self.hit_duration <= self.rec_rate/self.max_freq), \
		  f"hit_duration must be smaller than max_freq cycle duration."

		if self.hit_duration != 0:
			assert(self.hit_duration > 1/self.rec_rate), \
			  f"hit_duration must be greater than frame duration."

		assert (self.num_bands >= 0), \
		  f"number of bands must be 0 (continuous) or greater (discrete)."

		assert (self.rec_rate >= 2*self.max_freq), \
		  f"Max. frequency cannot be greater than half the recording rate."

		assert (self.gradient_space in ["linear", "log"]), \
		  f"gradient_space '{self.gradient_space}' not supported."

		assert (intensity <= 255) and (intensity >= 0), \
		  f"Intensity must be greater than or equal to 0 and less than or equal 255."

		# Sum of On and Off gives full cycle
		self.On_T = On_T
		self.Off_T = Off_T
		self.num_On_Off_frms = int((Off_T + On_T)*self.rec_rate*self.framepack_num)
		self.num_On_frms = int(On_T*self.rec_rate*self.framepack_num)

		# if num_bands = 0 then using continuous freq bands
		# set number of bands to be equal to arena size
		if self.num_bands == 0:
			if self.grad_dir in ["dx", "-dx"]:
				self.num_bands = int(self.config.getfloat("Projector", "res_dx"))
			else:
				self.num_bands = int(self.config.getfloat("Projector", "res_dy"))

		# build frequency bands mapping --
		# band_centers stores the center of each band
		# will compute distance from closest center
		if self.grad_dir in ["dx", "dy"]:
			self.band_centers = np.linspace(-1,
										    1,
											self.num_bands)
		else:
			# invert gradient direction
			self.band_centers = np.linspace(1,
											-1,
											self.num_bands)

		if self.gradient_space == "linear":
			self.frequency_values = np.linspace(self.min_freq,
											    self.max_freq,
											    self.num_bands)
		elif self.gradient_space == "log":
			self.frequency_values = np.geomspace(self.min_freq,
												 self.max_freq,
												 self.num_bands)

		# print(f"Setting frequency bands to: {self.band_centers}")
		# print(f"Setting frequency values to {self.frequency_values}")

		# frame counter used to flash stimuli
		self.freq_counter = np.zeros(self.max_flies)

		# max frame counter sets counter limit for a given frequency stimuli
		self.freq_counter_max = np.zeros(self.max_flies)

		# cycle counter stores converted frequency stimuli to frame number
		self.freq_cycle_counter = np.zeros(self.max_flies)

	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		each fly. Each box is red; just moved off screen if fly disappears.
		"""
		
		from psychopy import visual, filters
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.max_flies, elementMask='circle',
						xys=np.zeros((self.max_flies, 2)), 
						sizes=[self.ellipse_dx, self.ellipse_dy],
						colorSpace='rgb255', colors=(0,0,0),
						elementTex=None)

		if self.framepack_num == 3:
			self.shapes.colors = (self.intensity, self.intensity, self.intensity)
		else:
			self.shapes.colors = (self.intensity, 0, 0)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update stimuli to positions of moved flies.
		"""
		
		xs, ys = self.conv_cam_to_proj(centroids[:, 0], centroids[:, 1])

		# check gradient direction
		if self.grad_dir in ["dx", "-dx"]:
			coords = xs
		else:
			coords = ys

		if frm_num % int(self.rec_rate*self.update_interval) == 0:
			# only update freq stim after update_interval has passed
			
			# get band given position for all flies
			freq_band = np.abs(self.band_centers - coords[:, np.newaxis])
			freq_band = np.argmin(freq_band, axis=1)

			# get frequency given band for all flies
			freq_stim = self.frequency_values[freq_band]
			self.freq_cycle_counter = self.rec_rate//freq_stim
			if self.hit_duration > 0:
				self.freq_counter_max[states==1] = int(self.hit_duration*self.rec_rate)
			else:
				self.freq_counter_max = self.freq_cycle_counter

		# zero counters for lost flies
		# shouldn't need this, maybe remove if dropping frames
		self.freq_counter[states==0] = 0
		self.freq_counter_max[states==0] = 0
		self.freq_cycle_counter[states==0] = 0

		# increase frame counter for active flies
		self.freq_counter[states==1] += 1

		# if counter is double max, i.e. full cycle has passed, reset counter
		self.freq_counter[self.freq_counter>=(self.freq_cycle_counter*2)] = 0

		# if counter is less|equal than max, stimuli is on
		# otherwise, stimuli is off
		above_threshold = self.freq_counter > 0
		bellow_threshold = self.freq_counter <= self.freq_counter_max
		stim_on = (above_threshold & bellow_threshold).astype(int)

		# Frame in each On/Off block cycle
		block_frm = np.mod(frm_num, self.num_On_Off_frms)

		if self.grad_dir in ["dx", "-dx"]:
			xs = coords
		else:
			ys = coords

		if (block_frm < self.num_On_frms):
			xys = [xs + 3*(1 - states) + 3*(1 - stim_on), ys]
		else:
			xys = [xs + 6, ys]

		# Transpose list to (Nx2); faster than vstacking as numpy array
		vals = list(map(list, zip(*xys)))
		self.shapes.xys = vals


class intermittent_static_ribbon(stim_protocol):
	"""
	Single static ribbon horizontal to wind flow.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool
		If True, background is red and stimuli are black.
	width: float
		width of ribbon in mm.
	locs: list of floats
		Y-placements of ribbons in stimulus units (-1 to 1)
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, width=2, locs=[0, -0.5, 0.5],
				 On_T=15, Off_T=15, intensity=35):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		width: float
			Width of ribbon in mm
		locs: list of floats
			Y-placements of ribbons in stimulus units (-1 to 1)
		On_T, Off_T: int
			If both are not None, then bars are presented in blocks of
			On_T seconds, interrupted by Off_T seconds of no stimulus.
		intensity: int
			LED intensity for stimuli presentation
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
		conv_cam_to_proj = convert_units().conv_cam_to_proj
		mm_per_px = config.getfloat("Camera", "mm_per_px")
		stim_units_per_px = (conv_cam_to_proj(0, 1)[1] - 
							conv_cam_to_proj(0, 0)[1])

		self.width = width
		self.width_stim_units = width/mm_per_px*abs(stim_units_per_px)
		self.locs = locs

		self.framepack_num = int(2*framepack + 1)
		self.intensity = intensity

		# Sum of On and Off gives full cycle
		self.On_T = On_T
		self.Off_T = Off_T
		self.num_On_Off_frms = int((Off_T + On_T)*rec_rate*self.framepack_num)
		self.num_On_frms = int(On_T*rec_rate*self.framepack_num)
		
	def initialize_shapes(self):
		
		from psychopy import visual


		self.xys = np.vstack((np.zeros(len(self.locs)), self.locs)).T
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=len(self.locs), fieldShape='square',
						elementMask=None, xys=self.xys,
						sizes=[2, self.width_stim_units], 
						colorSpace='rgb255', colors=(0,0,0),
						elementTex=None)
		
		# self.shapes.opacities = np.prod(
		# 	, axis=0
		# 	)

		# mask_tex = visual.GratingStim(
		# 	win=self.win,
		# 	tex=None,
		# 	mask="gauss",
		# 	units="norm",
		# 	size=(1,1),
		# 	maskParams={'sd': 50},
		# 	)
		# mask_tex.AutoDraw = True
		# print(mask_tex.AutoDraw)

		# pixDeg = 20
		# dots = visual.DotStim(self.win, nDots = 500, coherence = 1, fieldPos=(0.0, 0.0), 
		#     fieldShape = 'circle', dotSize = 10 * pixDeg, dotLife = 5, dir=180, speed = 13.0/60 * pixDeg, colorSpace="rgb255", color=(255,0,0), 
		#     units = 'norm', opacity = 1.0, contrast = 1.0, depth = 0, signalDots = 'same', noiseDots = 'walk')
    
		# #maskParams
		# visibleArea = (14.7/2) * pixDeg
		# zeroOpacArea = (4.9 + 0.5) * pixDeg
		# shadedArea = (visibleArea - zeroOpacArea)/visibleArea

		# # # Create a raisedCosine mask array and assign it to a Grating stimulus (grey outside, transparent inside)
		# raisedCosTexture = visual.filters.makeMask(14.7 * pixDeg, shape= 'raisedCosine', fringeWidth= shadedArea, radius = [1.0, 1.0])
		# # # invRaisedCosTexture = -raisedCosTexture # inverts mask to blur edges instead of center
		# dotsMask = visual.GratingStim(
		# 	self.win,
		# 	mask="gauss",
		# 	tex=None,
		# 	contrast=1.0,
		# 	units="norm",
		# 	size=(1, 1),
		# 	colorSpace='rgb255',
		# 	color=(self.intensity,0,0)
		# 	)
		# dots.AutoDraw = True
		# dotsMask.AutoDraw = True
		
		if self.framepack_num == 3:
			self.shapes.colors = (self.intensity, self.intensity, self.intensity)
		else:
			self.shapes.colors = (self.intensity, 0, 0)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Display or hide ribbons depending on frame number.
		"""
		
		# Frame in each On/Off block cycle
		block_frm = np.mod(frm_num, self.num_On_Off_frms)

		if (block_frm < self.num_On_frms):
			xys = np.array([self.xys[:, 0], self.xys[:, 1]])
		else:
			xys = np.array([self.xys[:, 0] + 6, self.xys[:, 1]])

		# Transpose list to (Nx2); faster than vstacking as numpy array
		vals = list(map(list, zip(*xys)))
		self.shapes.xys = vals


class full_field_gradient(stim_protocol):
	"""
	Static gradient horizontal to wind flow.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool
		If True, background is red and stimuli are black.
	width: float
		width of ribbon in mm.
	locs: list of floats
		Y-placements of ribbons in stimulus units (-1 to 1)
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, min_intensity=0,
				 max_intensity=255, On_T=30, Off_T=30, grad_dir="dx",
				 grad_space="linear"):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		length: float
			Length of gradient in the arena (0 to 1). If 0.5, gradient will
		reach max intensity at 

		locs: list of floats
			Y-placements of ribbons in stimulus units (-1 to 1)
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
							
		self.grad_dir = grad_dir
		if self.grad_dir in ["dx", "-dx"]:
			stim_units_per_px = int(config.getfloat("Projector", "res_dx"))
		else:
			stim_units_per_px = int(config.getfloat("Projector", "res_dy"))
			
		# CHANGE THIS TO BE PERCENTAGE AND NOT MM/PIXEL
		self.stim_units = abs(stim_units_per_px)

		self.framepack_num = int(2*framepack + 1)
		self.min_intensity = min_intensity
		self.max_intensity = max_intensity

		# Sum of On and Off gives full cycle
		self.On_T = On_T
		self.Off_T = Off_T
		self.num_On_Off_frms = int((Off_T + On_T)*rec_rate*self.framepack_num)
		self.num_On_frms = int(On_T*rec_rate*self.framepack_num)

		# build gradient mask
		self.num_bands = int(2/self.stim_units)
		if self.grad_dir in ["dx", "-dy"]:
			self.band_centers = np.linspace(-1,
										    1,
											self.num_bands)
		else:
			# invert gradient direction
			self.band_centers = np.linspace(1,
											-1,
											self.num_bands)

		self.gradient_space = grad_space
		if self.gradient_space == "linear":
			self.intensity_values = np.linspace(self.min_intensity,
												self.max_intensity,
												self.num_bands)
		elif self.gradient_space == "log":
			self.intensity_values = np.geomspace(self.min_intensity,
												 self.max_intensity,
												 self.num_bands)
		# clip values
		self.intensity_values[self.intensity_values>255] = 255

	def initialize_shapes(self):
		
		from psychopy import visual

		if self.grad_dir in ["dx", "-dx"]:
			self.sizes = [self.stim_units, 2]
			self.xys = np.vstack((self.band_centers, np.zeros(self.num_bands))).T
		else:
			self.sizes = [2, self.stim_units]
			self.xys = np.vstack((np.zeros(self.num_bands), self.band_centers)).T

		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.num_bands, fieldShape='square',
						xys=self.xys, sizes=self.sizes, 
						colorSpace='rgb255', colors=(0, 0, 0),
						elementTex=None, elementMask=None,)

		if self.framepack_num == 3:
			color_list = np.array([(intensity, intensity, intensity) for intensity in self.intensity_values])
		else:
			color_list = np.array([(intensity, 0, 0) for intensity in self.intensity_values])
		self.shapes.colors = color_list
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Display or hide ribbons depending on frame number.
		"""
		
		# Frame in each On/Off block cycle
		block_frm = np.mod(frm_num, self.num_On_Off_frms)

		if (block_frm < self.num_On_frms):
			xys = np.array([self.xys[:, 0], self.xys[:, 1]])
		else:
			xys = np.array([self.xys[:, 0] + 6, self.xys[:, 1]])

		# Transpose list to (Nx2); faster than vstacking as numpy array
		vals = list(map(list, zip(*xys)))
		self.shapes.xys = vals


class sawtooth_gradient(stim_protocol):
	"""
	N static gradients over the whole arena.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	invert: bool
		If True, background is red and stimuli are black.
	width: float
		width of ribbon in mm.
	locs: list of floats
		Y-placements of ribbons in stimulus units (-1 to 1)
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, min_intensity=0, grad_repeats=3,
				 max_intensity=255, On_T=30, Off_T=30, grad_dir="dx",
				 grad_space="linear"):
		"""
		Constructor class for creating and manipulating psychopy objects.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		min_intensity: int
			lowest intensity value for the gradient
		max_intensity: int
			highest intensity value for the gradient
		On_T, Off_T: float
			If both are not None, then bars are presented in blocks of
			On_T seconds, interrupted by Off_T seconds of no stimulus.
		grad_dir: str
			gradient direction (±dx, ±dy)
		grad_space: str
			mapping function for the intensity values (linear, log)
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		config = load_settings()
				
		self.grad_dir = grad_dir
		if self.grad_dir in ["dx", "-dx"]:
			self.num_bands = int(config.getfloat("Projector", "res_dx"))
		else:
			self.num_bands = int(config.getfloat("Projector", "res_dy"))

		self.band_size = 2/self.num_bands
		
		self.framepack_num = int(2*framepack + 1)
		self.min_intensity = min_intensity
		self.max_intensity = max_intensity

		# Sum of On and Off gives full cycle
		self.On_T = On_T
		self.Off_T = Off_T
		self.num_On_Off_frms = int((Off_T + On_T)*rec_rate*self.framepack_num)
		self.num_On_frms = int(On_T*rec_rate*self.framepack_num)

		# build gradient mask
		if self.grad_dir in ["dx", "-dy"]:
			self.band_centers = np.linspace(-1,
										    1,
											self.num_bands)
		else:
			# invert gradient direction
			self.band_centers = np.linspace(1,
											-1,
											self.num_bands)

		self.grad_space = grad_space
		self.grad_repeats = grad_repeats

		if self.grad_space == "linear":
			self.intensity_values = np.linspace(self.min_intensity,
												self.max_intensity,
												self.num_bands//self.grad_repeats)
		elif self.grad_space == "log":
			self.intensity_values = np.geomspace(self.min_intensity,
												 self.max_intensity,
												 self.num_bands//self.grad_repeats)
		self.intensity_values = np.tile(self.intensity_values, self.grad_repeats)
		
		# clip values
		self.intensity_values[self.intensity_values>255] = 255

	def initialize_shapes(self):
		
		from psychopy import visual

		if self.grad_dir in ["dx", "-dx"]:
			self.sizes = [self.band_size, 2]
			self.xys = np.vstack((self.band_centers, np.zeros(self.num_bands))).T
		else:
			self.sizes = [2, self.band_size]
			self.xys = np.vstack((np.zeros(self.num_bands), self.band_centers)).T

		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.num_bands, fieldShape='square',
						xys=self.xys, sizes=self.sizes, 
						colorSpace='rgb255', colors=(0, 0, 0),
						elementTex=None, elementMask=None,)
		
		if self.framepack_num == 3:
			color_list = np.array([(intensity, intensity, intensity) for intensity in self.intensity_values])
		else:
			color_list = np.array([(intensity, 0, 0) for intensity in self.intensity_values])
		
		self.shapes.colors = color_list
		self.shapes.setAutoDraw(True)

	def update(self, ct, centroids, states, frm_num):
		"""
		Display or hide ribbons depending on frame number.
		"""
		
		# Frame in each On/Off block cycle
		block_frm = np.mod(frm_num, self.num_On_Off_frms)

		if (block_frm < self.num_On_frms):
			xys = np.array([self.xys[:, 0], self.xys[:, 1]])
		else:
			xys = np.array([self.xys[:, 0] + 6, self.xys[:, 1]])

		# Transpose list to (Nx2); faster than vstacking as numpy array
		vals = list(map(list, zip(*xys)))
		self.shapes.xys = vals


class static_ribbon_individual_flies(stim_protocol):
	"""
	Position-based red circle flashes the fly given its position.
	
	Attrs
	-------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	conv_cam_to_proj: conv_cam_to_proj method of convert_units class
		Converts units from camera pixels to stimulus units.
	ellipse_dx: float
		Horizontal size of circle in stimulus units.
	ellipse_dy: float
		Vertical size of circle in stimulus units.
	"""
		
	def __init__(self, win, rec_rate, max_flies, invert=False, 
				 framepack=False, intensity=35, circle_diam=10,
				 width=2, locs=[0, -0.5, 0.5]):
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
	
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid
		invert: bool;
			If True, background is red and stimuli are black.
		framepack: bool
			If True, pack 3 frames into single 16.67 ms flip
		intensity: int
			The rgb255 intensity for the red light displayed on the projector.
		circle_diam: float
			Diameter of stimulus circle covering each fly, in mm.
		width: float
			width of ribbon in mm.
		locs: list of floats
			Y-placements of ribbons in stimulus units (-1 to 1)
		"""
		
		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		self.config = load_settings()
		self.conv_cam_to_proj = convert_units().conv_cam_to_proj
		mm_per_px = self.config.getfloat("Camera", "mm_per_px")

		stim_dx_per_px = (self.conv_cam_to_proj(1, 0)[0] - 
							self.conv_cam_to_proj(0, 0)[0])
		stim_dy_per_px = (self.conv_cam_to_proj(0, 1)[1] - 
							self.conv_cam_to_proj(0, 0)[1])

		self.ellipse_dx = circle_diam/mm_per_px*stim_dx_per_px
		self.ellipse_dy = circle_diam/mm_per_px*stim_dy_per_px

		self.framepack_num = int(2*framepack + 1)
		self.intensity = intensity

		self.width = width
		self.width_stim_units = width/mm_per_px*abs(stim_dy_per_px)
		self.locs = locs

		assert (intensity <= 255) and (intensity >= 0), \
		  f"Intensity must be greater than or equal to 0 and less than or equal 255."

	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		each fly. Each box is red; just moved off screen if fly disappears.
		"""
		
		from psychopy import visual, filters
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=self.max_flies, elementMask='circle',
						xys=np.zeros((self.max_flies, 2)), 
						sizes=[self.ellipse_dx, self.ellipse_dy],
						colorSpace='rgb255', colors=(0,0,0),
						elementTex=None)

		if self.framepack_num == 3:
			self.shapes.colors = (self.intensity, self.intensity, self.intensity)
		else:
			self.shapes.colors = (self.intensity, 0, 0)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update stimuli to positions of moved flies.
		"""
		
		xs, ys = self.conv_cam_to_proj(centroids[:, 0], centroids[:, 1])
		
		# repeat ys for each ribbon locations
		ext_ys = np.vstack([ys for _ in self.locs]).T
		
		# subtract ys from each ribbon location + half their width
		sub_ext_ys = np.abs(ext_ys - self.locs) - self.width_stim_units/2
		
		# find negative numbers (these flies are within a ribbon)
		ribbon_flies = sub_ext_ys < 0
		# if fly is within a ribbon, sum should be positive
		stim_on = np.sum(ribbon_flies, axis=1) > 0
		
		xys = [xs + 3*(1 - states) + 3*(1 - stim_on), ys]

		# Transpose list to (Nx2); faster than vstacking as numpy array
		vals = list(map(list, zip(*xys)))
		self.shapes.xys = vals


###############################################################################
#################               Deprecated                     ################
###############################################################################


class full_field_flash_spec(stim_protocol):
	"""
	CANNOT YET HANDLE FRAME PACKING
	
	Flash entire arena at a given frequency. Flash lasts for defined duration, 
	either in percentage or number of frames.
	
	Attrs
	--------
	
	win: psychopy window object
		Projector for stimulus.
	rec_rate: float
		Recording frame rate in fps
	max_flies: int
		Maximum number of fly trajectories can that can be recorded in video.
	invert: bool;
		If True, background is red and stimuli are black.
	framepack: bool
		If True, pack 3 frames into single 16.67 ms flip
	freq: float
		frequency of flashes in flash/sec
	shapes: psychopy.visual.ElementArrayStim object instance
		Container for all shapes that will be projected on each flip.
	frms_per_flash: int
		Number of frames for one flash cycle.
	"""
	
	def __init__(self, win, rec_rate, max_flies, invert, freq=5, 
				 stim_len=5, dur=0.5, framepack=False):
		
		"""
		Constructor class for stimuli projection.
		
		Args
		-------
		win: psychopy window object
			Projector for stimulus.
		rec_rate: float
			Recording frame rate in fps.
		max_flies: int
			Maximum number of fly trajectories can that can be recorded in vid.
		invert: bool;
			If True, background is red and stimuli are black.
		framepack_num: int
			3 if packing RGB frames; 1 otherwise
		stim_freq: float
			Frequency of flashes in frames/sec. Best to keep it a 
			fraction of half the recording rate.
		stim_len: time during which stimulus is delivered, in seconds
		### note the stim_len of the ON and OFF blocks is always the same
		dur: float
			Duration of flash in number of frames OR in percentage      
		"""

		super().__init__(win, rec_rate, max_flies, invert, framepack)
		
		self.freq = freq
		self.stim_len = stim_len
		self.dur = dur
		assert self.rec_rate >= 2*freq, \
			'Flash rate cannot be greater than half the recording rate.'
		self.framepack_num = int(2*framepack + 1)
		
		## how many frames is the ON block
		self.stim_len_frms = self.stim_len*self.rec_rate*self.framepack_num

		## how many frames per flash
		self.max_frms_per_flash = int(self.rec_rate/self.freq/
									self.framepack_num)

		## using dur as percentage
		if dur < 1:
			self.frms_per_flash = int(self.max_frms_per_flash*self.dur)		
		else:
		## using dur as number of MILLISECONDS
			self.dur_frms = self.dur/1000*self.rec_rate*self.framepack_num
			assert self.dur_frms < self.max_frms_per_flash, \
				'Duration cannot be longer than the maximum number of frames'\
				'per flash \n i.e. frame rate/frequency' 
			self.frms_per_flash = int(self.dur_frms)
		
	def initialize_shapes(self):
		"""
		Constructor class for stimuli projection. Draw single red box on 
		whole screen.
		"""
		
		from psychopy import visual
		self.shapes = visual.ElementArrayStim(self.win, 
						nElements=1, elementMask=None,
						xys=[[0, 0]], fieldShape='sqr',
						sizes=[[2, 2]], colors=self.shapes_color, 
						elementTex=None)
		self.shapes.setAutoDraw(True)
		
	def update(self, ct, centroids, states, frm_num):
		"""
		Update position if on frame number to update. Make black by moving box
		too far to right, off screen.
		"""
		
		## 5 s, divisible by frames per flash, move off screen for gap length
		## find divisions for each gap and freq length
		
		if (frm_num == 0):
			statusB = 0 ## moving on and off screen for cycle

	   ## divide frame number into the stim/gap length, 
	   ### if odd, stim should be on screen
	   ### if even, stim should be off screen
		if(int(frm_num/self.stim_len_frms) % 2 == 0):
			statusB = 1
		else:
			statusB = 0
		   
		if(statusB == 0):
			self.shapes.xys = [[3, 0]]
	   
		if(statusB == 1):
			if(frm_num % self.max_frms_per_flash == 0):
				self.shapes.xys = [[0, 0]]
				 
			if(frm_num % self.max_frms_per_flash == self.frms_per_flash):
				self.shapes.xys = [[3, 0]]
