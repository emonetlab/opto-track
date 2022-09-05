

# WITH WIND
# PATTERN MODE


import sys
sys.path.append('../src')
from exp_funcs import rec_and_proj
from stimuli import *
import time


speeds = [-30,
			30,
			-30,
			-20,
			30,
			20,
			-30,
			20,
			30,
			20,
			-20,
			30,
			-20,
			30,
			20,
			-30,
			-30,
			-20,
			20,
			-20
			]
		

speed = speeds[int(sys.argv[1])]

a = rec_and_proj(stimulus_protocol=horiz_moving_bars, 
				 rec_length=60, 
				 init_T=2,
				 stimulus_protocol_kwargs=
					({
					'invert': False,
					'speed': speed,
					'on_width': abs(float(speed*2)),
					}),
				exp_name='moving_bars_plus_wind_perp_speed=%d' % speed,
				framepack=False,
				bypass_cam=False,
				disp_frm_skips_lens=False
				)
				
a.run()

print ('Waiting %d seconds for next experiment...' % exp_delay)
time.sleep(60)

quit()