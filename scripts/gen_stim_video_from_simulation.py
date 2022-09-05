"""
Generate a stimulus video from a recorded smoke video. 

Created by Nirag Kadakia at 23:10 8-23-2021
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
sys.path.append('../src')
from load_save_data import load_settings, save_stim_video_metadata, \
                           get_stim_vids_dir
						   
						 
filename = 'D:/odor-plume-sim/1mm_grid/conc.npz'
conc_grid = np.load(filename)['arr_0']
conc_grid = np.reshape(conc_grid, (360, 640, conc_grid.shape[1]), order='C')
sim_vid_fps = 100
sim_vid_mm_per_px = 0.5	

settings = load_settings()
proj_mm_per_px_x = load_settings().getfloat("Projector", "mm_per_px_x")
proj_mm_per_px_y = load_settings().getfloat("Projector", "mm_per_px_y")

# Threshold
min_val = 0
stim_mult = 10000

# Reverse video?
reverse = False


# Frames to load, smoothing pixels, desired frame rate, name
frms_to_load = range(500, 2000)

fps = 180
stim_name = 'simulation_1'

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

    data = conc_grid[:, :, frm].astype(np.float64)
    data = data.T
    
    # Save at pix-mm resolution of the projector -- maps mm to mm faithfully
    px_x = 1/proj_mm_per_px_x*sim_vid_mm_per_px
    px_y = 1/proj_mm_per_px_y*sim_vid_mm_per_px

    M = np.array([[px_y, 0., 0.], 
                 [0., px_x, 0.]])
    data = cv2.warpAffine(data, M, 
        (int(px_y*data.shape[1]), 
         int(px_x*data.shape[0])))

    # Threshold and get it in the correct integer format
    thresh_data = np.maximum(np.minimum((
                    data - min_val)*stim_mult, 255), 0)
    thresh_data = thresh_data.astype('uint8').T
    
    # Color is BGR format in opencv -- save as red
    if fps == 60:
        bg_channels = np.zeros(thresh_data.shape).astype('uint8')
    elif fps == 180:
        bg_channels = thresh_data
    color_data = np.asarray([bg_channels.T, bg_channels.T, thresh_data.T]).T

    if iF == 0:
        out = cv2.VideoWriter('tmp_%s.avi' % file_name, 
                cv2.VideoWriter_fourcc(*'XVID'), sim_vid_fps, 
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
metadata['sim_file_name'] = filename
metadata['proj_mm_per_px_x'] = proj_mm_per_px_x
metadata['proj_mm_per_px_y'] = proj_mm_per_px_y
metadata['fps'] = fps
metadata['size'] = thresh_data.T.shape
metadata['num_frames'] = len(frms_to_load)

save_stim_video_metadata(metadata, file_name)