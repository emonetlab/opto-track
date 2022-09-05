"""
Script to gather data in an optogenetic experiment. 
Created by Nirag Kadakia at 15:08 06-27-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import os
import sys
sys.path.append(os.path.join('..', 'src'))
from exp_funcs import rec_and_proj
from stimuli import *

a = rec_and_proj(stimulus_protocol=static_ribbon, 
				 rec_length=2, 
				 pre_rec_length=1,
				 init_T=1,
				 stimulus_protocol_kwargs=(
						{
						}
					),
				exp_name='test',
				run_exp_filepath=os.path.realpath(__file__),
				disp_frm_skips_lens=False,
				framepack=False,
				bypass_cam=True
				)
				
a.run()
