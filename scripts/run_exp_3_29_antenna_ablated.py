

# WITH WIND
# PATTERN MODE


import sys
sys.path.append('../src')
from exp_funcs import rec_and_proj
from stimuli import *
import time


directions = [-1, 
		1, 
		-1,
		-1, 
		1, 
		1, 
		-1, 
		1, 
		-1, 
		1,
		1, 
		-1,
		-1, 
		1, 
		1, 
		-1, 
		1, 
		-1, 
		1]

t_steps = [3,
			3,
			3,
			4,
			4,
			3,
			4,
			4,
			3,
			4,
			3,
			3,
			3,
			4,
			4,
			3,
			4,
			4,
			3,
			4
		]
			

t_step = t_steps[int(sys.argv[1])]
direction = directions[int(sys.argv[1])]

a = rec_and_proj(stimulus_protocol=scintillator_2, 
				 rec_length=60, 
				 init_T=3,
				 stimulus_protocol_kwargs=
					({
					'invert': False,
					't_step': t_step,
					'corr_sign': 1,
					'corr_x_step': 1,
					'direction': direction,
					'switch_T': 8,
					'motion_T': 4,
					'num_corrs': 1,
					'px_per_bar': 1,
					'static_motion_off': True
					}),
				exp_name='ant_ablated_scintillator_t_step=%d_dir=%s' % (t_step, direction),
				framepack=True,
				bypass_cam=False,
				disp_frm_skips_lens=False
				)
				
a.run()

print ('Waiting %d seconds for next experiment...' % exp_delay)
time.sleep(60)

quit()