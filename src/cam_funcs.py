"""
Functions for loading Point Grey Camera using FlyCapture2 
python module, getting settings, grabbing frames, and setting 
the viewable region.

Created by Nirag Kadakia at 13:52 01-03-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
import matplotlib.pyplot as plt
import PyCapture2 
from load_save_data import load_settings, save_settings


def print_build_info():
	"""
	Print the build info of the PyCapture2 library
	"""
	
	lib_ver = PyCapture2.getLibraryVersion()
	print("PyCapture2 library version: ", lib_ver[0], lib_ver[1], 
		   lib_ver[2], lib_ver[3])
	
def print_camera_info(cam):
	"""
	Print the info of the Point Grey Camera
	
	Args
	-------
	
	cam: PyCapture2.camera() object
		Camera object
	"""
	
	cam_info = cam.getCameraInfo()
	print("\n*** CAMERA INFORMATION ***\n")
	print("Serial number - ", cam_info.serialNumber)
	print("Camera model - ", cam_info.modelName)
	print("Camera vendor - ", cam_info.vendorName)
	print("Sensor - ", cam_info.sensorInfo)
	print("Resolution - ", cam_info.sensorResolution)
	print("Firmware version - ", cam_info.firmwareVersion)
	print("Firmware build time - ", cam_info.firmwareBuildTime)
	
def print_format7_capabilities(fmt7info):
	"""
	Print the capabilities for format7
	
	Args
	-------
	
	fmt7info: object instance
		Format of PyCapture2 format7 
	"""
	
	print("Max image pixels: ({}, {})".format(
		   fmt7info.maxWidth, fmt7info.maxHeight))
	print("Image unit size: ({}, {})".format(
		   fmt7info.imageHStepSize, fmt7info.imageVStepSize))
	print("Offset unit size: ({}, {})".format(
		   fmt7info.offsetHStepSize, fmt7info.offsetVStepSize))
	print("Pixel format bitfield: 0x{}".format(
		   fmt7info.pixelFormatBitField))
	print
	
def grab_imgs(cam, num_imgs_to_grab):
	"""
	Record a single or many consecutive frames from Point Grey camera.
	
	Args
	-------
	
	cam: PyCapture2.camera() object
		Camera object
	num_imgs_to_grab: int
		Number of images to record at frame rate.

	Returns
	-------
	
	cv_image: numpy array
		recorded image array
	"""
	
	prevts = None
	for iN in range(num_imgs_to_grab):
		try:
			image = cam.retrieveBuffer()
		except PyCapture2.Fc2error as fc2Err:
			print("Error retrieving buffer : ", fc2Err)
			continue

		ts = image.getTimeStamp()
		if (prevts):
			diff = (ts.cycleSeconds - prevts.cycleSeconds)*8000 \
				   + (ts.cycleCount - prevts.cycleCount)
			print("Timestamp [", ts.cycleSeconds, ts.cycleCount, "] -", diff)
		prevts = ts

	row_bytes = float(len(image.getData()))/float(image.getRows())
	cv_image = np.array(image.getData(), dtype="uint8").reshape(
				(image.getRows(), image.getCols()))
	
	return cv_image

def detect_cam_fps(c):
	"""
	Detect the frame rate of the camera.
	
	Args
	-------
	
	c: PyCapture2.camera() object
		Camera object.
		
	Returns
	-------
		
	frame_rate: float
		Recording frame rate in frames per second.
	"""
	
	frame_rate_prop = c.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE)
	frame_rate = frame_rate_prop.absValue
	print("Using frame rate of {}".format(frame_rate))
	
	return frame_rate

def init_cam(x0=0, y0=50, dx=2048, dy=1100):
	"""
	Initialize the camera and field of view size.
	
	Args
	-------
	
	x0, y0: ints
		limits of left and bottom camera field of view.
	dx, dy: ints
		width and height of camera field of view.
	
	Returns
	-------
		
	c: PyCapture2.camera() object
		Camera object.
	"""
			
	print_build_info()
	bus = PyCapture2.BusManager()
	num_cams = bus.getNumOfCameras()
	print("Number of cameras detected: ", num_cams)
	if not num_cams:
		print("Insufficient number of cameras. Exiting...")
		exit()

	# Select camera on 0th index
	c = PyCapture2.Camera()
	uid = bus.getCameraFromIndex(0)
	c.connect(uid)

	print_camera_info(c)
	
	fmt7_info, supported = c.getFormat7Info(0)
	print_format7_capabilities(fmt7_info)

	print('Step sizes:')
	print('x0:', fmt7_info.offsetHStepSize)
	print('y0:', fmt7_info.offsetVStepSize)
	print('dx:', fmt7_info.imageHStepSize)
	print('dy:', fmt7_info.imageVStepSize)
	print('Maximums:')
	print('dx:', fmt7_info.maxWidth)
	print('dy:', fmt7_info.maxHeight)
	
	# Check whether pixel format mono8 is supported
	if PyCapture2.PIXEL_FORMAT.MONO8 & fmt7_info.pixelFormatBitField == 0:
		print("Pixel format is not supported\n")
		exit()
	
	# Configure format7 settings / or: fmt7_info.maxWidth, or fm7info.maxheight
	c = set_cam_region(c, x0, y0, dx, dy)
	
	return c
	
def set_cam_region(c, x0, y0, dx, dy):
	"""
	Initialize the camera and field of view size.
	
	Args
	-------
	
	x0, y0: ints
		limits of left and bottom camera field of view.
	dx, dy: ints
		width and height of camera field of view.
	
	Returns
	-------
		
	c: PyCapture2.camera() object
		Camera object.
	"""
	
	fmt7_img_set = PyCapture2.Format7ImageSettings(0, x0, y0, dx, dy, 
												 PyCapture2.PIXEL_FORMAT.MONO8)
	fmt7_pkt_info, fmt7_is_valid = c.validateFormat7Settings(fmt7_img_set)
	if not fmt7_is_valid:
		print("Format7 settings are not valid!")
		exit()
	c.setFormat7ConfigurationPacket(fmt7_pkt_info.recommendedBytesPerPacket, 
									fmt7_img_set)
	
	return c
	
	
class calibrate_cam():
	"""
	Class for setting camera viewable region 
	"""
	
	def __init__(self):
		"""
		Initialize camera and plotting window.
		"""
		
		self.config = load_settings()

		self.x0 = self.config.getfloat("Camera", "x0")
		self.y0 = self.config.getfloat("Camera", "y0")
		self.dx = self.config.getfloat("Camera", "dx")
		self.dy = self.config.getfloat("Camera", "dy")
		
		self.x0_step = self.config.getfloat("Camera", "x0_step")
		self.y0_step = self.config.getfloat("Camera", "y0_step")
		self.dx_step = self.config.getfloat("Camera", "dx_step")
		self.dy_step = self.config.getfloat("Camera", "dy_step")
		
		self.max_width = self.config.getfloat("Camera", "max_width")
		self.max_height = self.config.getfloat("Camera", "max_height")
		
		# Initialize camera
		self.c = init_cam(self.x0, self.y0, self.dx, self.dy)
		self.c.startCapture()
		
		# Set up image window
		plt.ion()
		self.fig = plt.figure()
		self.ax = plt.imshow(grab_imgs(self.c, 1))
		plt.show()
		plt.draw()
		self.fig.canvas.flush_events()
		
	def update_camera_and_plotting_window(self):
		"""
		Update camera and plotting window
		"""
		
		self.c.stopCapture()
		self.dx = max(min(self.dx, self.max_width), 0)
		self.dy = max(min(self.dy, self.max_height), 0)
		self.x0 = max(min(self.x0, self.max_width), 0)
		self.y0 = max(min(self.y0, self.max_height), 0)
		self.c = set_cam_region(self.c, self.x0, self.y0, self.dx, self.dy)
		self.c.startCapture()
		
		self.ax.set_data(grab_imgs(self.c, 1))
		plt.draw()
		self.fig.canvas.flush_events()
		
	def run(self):
		"""
		Run the calibration. Accepts user input continuously until "Enter".
		
		d-f and D-F change y0 and dy respectively, by 
		their steps given by the camera setting units respectively. 
		j-k and J-K do the same for x0 and dx.
		"""
		
		self.update_camera_and_plotting_window()
		
		while True:
			
			print(self.x0, self.y0, self.dx, self.dy)
			
			keys = input()
			try: 
				keys = keys.decode('ASCII')
			except (UnicodeDecodeError):
				print('invalid ASCII key')
				
			if keys == 'k':
				self.dx += self.dx_step
			elif keys == 'j':
				self.dx -= self.dx_step
			
			elif keys == 'd':
				self.x0 -= self.x0_step
			elif keys == 'f':
				self.x0 += self.x0_step
			
			elif keys == 'D':
				self.y0 -= self.y0_step
			elif keys == 'F':
				self.y0 += self.y0_step
			
			elif keys == 'K':
				self.dy += self.dy_step
			elif keys == 'J':
				self.dy -= self.dy_step

			elif keys == '\r':
				break
			
			self.update_camera_and_plotting_window()
			
		self.config.set("Camera", "x0", '%.1f' % self.x0)
		self.config.set("Camera", "y0", '%.1f' % self.y0)
		self.config.set("Camera", "dx", '%.1f' % self.dx)
		self.config.set("Camera", "dy", '%.1f' % self.dy)
		
		plt.close()
		plt.ioff()
		
		save_settings(self.config)
		self.c.stopCapture()
