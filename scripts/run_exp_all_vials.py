import sys
sys.path.append('../src')
from exp_funcs import rec_and_proj
from stimuli import *
import time

# Time between experiments
exp_delay = 60

vals = [
[1 , -1],
[1 , 1],
[-1 , 1],
[-1 , -1],
[1 , -1],
[1 , 1],
[-1 , 1],
[-1 , -1],
[1 , -1],
[1 , 1]
]

vals_to_run = vals[int(sys.argv[1])]

a = rec_and_proj(stimulus_protocol=scintillator_2,
				rec_length=60,
				stimulus_protocol_kwargs=(
					{'corr_sign': vals_to_run[1],
					 'direction': vals_to_run[0],
					 'num_corrs': 1,
					 'motion_T': 4, 
					 'switch_T': 8,
					 't_step': 3,
					 'static_motion_off': True
					}
				),
				framepack=True,
				init_T=4,
				exp_name='scintillator_t_step=3_no_wind_dir=%d_corr=%d' % (vals_to_run[0], vals_to_run[1]),
				disp_frm_skips_lens=False
				)
a.run()

print ('Waiting %d seconds for next experiment...' % exp_delay)
time.sleep(exp_delay)

quit()