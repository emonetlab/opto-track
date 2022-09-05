"""
Generate the data from a recording.

Created by Nirag Kadakia at 15:07 06-27-2019
This work is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License.
To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gen_fly_data_from_vid import gen_xy_theta, \
								  smooth_data, \
								  gen_stim_by_traj, \
								  get_fly_properties

folder = str(sys.argv[1])

# Flags are no-smooth, no-stim, or no-behaving to skip those routines
flags = []
if any(sys.argv[2:]):
	flags = str(sys.argv[2:])

a = gen_xy_theta(subdir=folder)
a.gen_data_all_exp_dirs()

if 'no-smooth' in flags:
	pass
else:
	b = smooth_data(subdir=folder)
	b.smooth_xy_all_exp_dirs()
	b.smooth_theta_all_exp_dirs()

if 'no-stim' in flags:
	pass
else:
	c = gen_stim_by_traj(subdir=folder)
	c.gen_stim_all_exp_dirs()

if 'no-behaving' in flags:
	pass
else:
	d = get_fly_properties(subdir=folder)
	d.get_behaving_all_exp_dirs(min_dist=20, min_displ=0)
