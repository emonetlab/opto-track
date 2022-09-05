"""
Generate a stimulus video from a recorded smoke video. 

Created by Nirag Kadakia at 15:10 10-03-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
import os
import shutil
import pickle
import cv2
import sys
import gzip
sys.path.append('../src')
from load_save_data import load_metadata_mat, load_vid_by_frm, \
						   load_settings, save_stim_video_metadata, \
						   get_stim_vids_dir

# Boundary regions to zero-out from recording
ymin = 230
ymax = 1150
xmin = 170
xmax = 1960

rec_video_metadata = load_metadata_mat()
settings = load_settings()
smoke_vid_fps = rec_video_metadata['p']['fps'][0]
vid_mm_per_px = rec_video_metadata['p']['mm_per_px'][0][0]
proj_mm_per_px_x = settings.getfloat("Projector", "mm_per_px_x")
proj_mm_per_px_y = settings.getfloat("Projector", "mm_per_px_y")

# Threshold
min_val = 3
stim_mult = 10

# Reverse video?
reverse = True

# Frames to load, smoothing pixels, desired frame rate, name
frms_to_load = range(5400)

num_smth_px = 2
fps = 180
stim_name = 'IS_bck_sub'

# Where video exists in Mahmut/Data2 directory
subdir = '2018_09_12 IS ISP SR Smoke Only'
file = '2018_09_12_NA_3_3ds_5do_IS_1'
#file = '2018_09_12_NA_2_3ds_5do_ISPOn_1'

if fps == 180:
	color = 'white'
elif fps == 60:
	color = 'red'
else:
	print('need fps of 180 or 60')
	quit()

if reverse == False:
	file_name = stim_name
else:
	frms_to_load = frms_to_load[::-1]
	file_name = stim_name + '_reverse'

for iF, frm in enumerate(frms_to_load):
	print(iF)
	
	data = load_vid_by_frm(subdir=subdir, file=file, frame=frm, bck_sub=True)
	data = data.astype(np.float64)
	
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
metadata['original_recording_file'] = file
metadata['proj_mm_per_px_x'] = proj_mm_per_px_x
metadata['proj_mm_per_px_y'] = proj_mm_per_px_y
metadata['fps'] = fps
metadata['size'] = thresh_data.T.shape
metadata['num_frames'] = len(frms_to_load)

save_stim_video_metadata(metadata, file_name)