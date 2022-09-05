"""
Script to gather data in an optogenetic experiment. 

Created by Nirag Kadakia at 15:08 06-27-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
sys.path.append('../src')
from exp_funcs import rec_and_proj
from stimuli import *
import time


exp_idx = int(sys.argv[1])
exp_delay = 60

vals_to_run = [0, 
1, 
1, 
0, 
1, 
1, 
0, 
1, 
0,
0
]

if vals_to_run[exp_idx] == 0:
	stim = 'new_smoke_1a'
else:
	stim = 'new_smoke_2a'
	
a = rec_and_proj(stimulus_protocol=movie, 
				 rec_length=60, 
				 init_T=1,
				 stimulus_protocol_kwargs=
					({
					'stim_name': stim,
					'codec': 'avi',
					'loop': True,
					'intensity': 1,
					'flip_horiz': True,
					}),
				exp_name=stim + 'vial1',
				framepack=True,
				bypass_cam=True,
				disp_frm_skips_lens=False
				)
				
a.run()
time.sleep(exp_delay)
quit()
