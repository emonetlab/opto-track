"""
Script to run functions before initializing experiment: setting viewable area, 
calibration. 

Created by Nirag Kadakia at 15:07 06-27-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
sys.path.append('../src')
from cam_funcs import calibrate_cam
from proj_funcs import calibrate_proj

a = calibrate_cam()
a.run()
a = calibrate_proj()
a.run()

