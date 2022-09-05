"""
Generate a stimulus video from a recorded smoke video. 

Created by Nirag Kadakia at 12:10 03-06-2022
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import cv2
import sys
import gzip
sys.path.append('../src')
from load_save_data import load_rec_vid_frame, load_settings, \
						   save_stim_video_metadata, get_stim_vids_dir

# Boundary regions to zero-out from recording
ymin = 0
ymax = 1088
xmin = 0
xmax = 1728

settings = load_settings()
smoke_vid_fps = 60
vid_mm_per_px = settings.getfloat("Camera", "mm_per_px")
proj_mm_per_px_x = settings.getfloat("Projector", "mm_per_px_x")
proj_mm_per_px_y = settings.getfloat("Projector", "mm_per_px_y")

# Plot to visualize
plot = 0

# Threshold
# new_smoke_1a (rate=500)
#min_val = 14
#stim_mult = 25
min_val = 10
stim_mult = 40

# Background frame
bck_frm = 15
bck_frm = 120
bck_data = None

# Reverse video?
reverse = True

# Invert to make smoke opaque?
invert = False

# Frames to load, smoothing pixels, desired frame rate, name
frms_to_load = range(400, 1800)

num_smth_px = 8
fps = 180
stim_name = 'new_smoke_2a'
#stim_name = 'vid_for_pres'

# Where video exists in Mahmut/Data2 directory
subdir = '2022-3-4'
#exp_name = 'jet_test_2wick_PSI=50_odor=100_switch_T=0.1_jet_rate=500_seed=1_1'
#exp_name = 'jet_test_2wick_PSI=55_odor=100_switch_T=0.1_jet_rate=400_seed=1_0'
exp_name = 'jet_test_2wick_PSI=40_odor=100_switch_T=0.1_jet_rate=700_seed=1_1'
#exp_name = 'jet_test_2wick_PSI=40_odor=100_switch_T=0.1_jet_rate=700_seed=1_0'

if fps == 180:
	color = 'white'
elif fps == 60:
	color = 'red'
else:
	print ('need fps of 180 or 60')
	quit()

if reverse == False:
	file_name = stim_name
else:
	frms_to_load = frms_to_load[::-1]
	file_name = stim_name + '_reverse'

for iF, frm in enumerate(frms_to_load):
	print (iF)
	
	data = load_rec_vid_frame(subdir, exp_name, frm)
	if bck_data is None:
		bck_data = load_rec_vid_frame(subdir, exp_name, bck_frm)
	
	data = np.maximum(data.astype(np.float32) - 
									bck_data.astype(np.float32), 
									np.zeros(data.shape)).astype(np.int8)
	data = data[:, :, 0]
	data = data.T
	
	data = data.astype(np.float32)
	kernel = np.ones((num_smth_px, num_smth_px), np.float32)/num_smth_px**2
	smoothed_data = cv2.filter2D(data, -1, kernel)
	
	smoothed_data[:xmin, :] = 0
	smoothed_data[:, :ymin] = 0
	smoothed_data[xmax:, :] = 0
	smoothed_data[:, ymax:] = 0
	
	# Save at pix-mm resolution of the projector -- maps mm to mm faithfully
	px_x = 1/proj_mm_per_px_x*vid_mm_per_px
	px_y = 1/proj_mm_per_px_y*vid_mm_per_px
	
	M = np.array([[px_y, 0., 0.], 
				 [0., px_x, 0.]])
	smoothed_data = cv2.warpAffine(smoothed_data, M, 
		(int(px_y*smoothed_data.shape[1]), 
		 int(px_x*smoothed_data.shape[0])))
	
	# Threshold and get it in the correct integer format
	thresh_data = np.maximum(np.minimum((
				    smoothed_data - min_val)*stim_mult, 255), 0)
	thresh_data = thresh_data.astype('uint8').T
	
	# Color is BGR format in opencv -- save as red
	if fps == 60:
		bg_channels = np.zeros(thresh_data.shape).astype('uint8')
	elif fps == 180:
		bg_channels = thresh_data
	color_data = np.asarray([bg_channels.T, bg_channels.T, thresh_data.T]).T
	
	if invert == True:
		color_data = 255 - color_data
	
	if plot == True:
		plt.imshow(color_data, vmin=0, vmax=255)
		plt.show()
	
	if iF == 0:
		out = cv2.VideoWriter('tmp_%s.avi' % file_name, 
				cv2.VideoWriter_fourcc(*'XVID'), smoke_vid_fps, 
				thresh_data.T.shape)
	out.write(color_data)
out.release()

# Adjust fps rate using ffmpeg; then save
os.system("ffmpeg -i tmp_%s.avi -filter:v fps=%s %s.avi" %
		(file_name, fps, file_name))
shutil.move('%s.avi' % file_name, '%s/%s.avi' 
			% (get_stim_vids_dir(), file_name))
os.remove('tmp_%s.avi' % file_name)

# Save information about the video
metadata = dict()
metadata['original_recording_subdir'] = subdir
metadata['original_recording_exp_name'] = exp_name
metadata['proj_mm_per_px_x'] = proj_mm_per_px_x
metadata['proj_mm_per_px_y'] = proj_mm_per_px_y
metadata['fps'] = fps
metadata['size'] = thresh_data.T.shape
metadata['num_frames'] = len(frms_to_load)

save_stim_video_metadata(metadata, file_name)