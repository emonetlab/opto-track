"""
Generate fly events (stops, turns, etc.) from x, y, and stimuli that has
already been generated from the video.

Created by Nirag Kadakia at 10:28 09-20-2019
This work fly_num licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
from glob import glob
from load_save_data import load_exp_data, load_exp_matrix, save_exp_matrix, \
						   save_metadata, load_metadata, get_data_dir
from utils import get_traj_ranges
import warnings

def calc_thresholded_events(arr, fps, traj_ranges, on_thresh, off_thresh,
							min_on_dur, min_off_dur):
	"""
	Calculate continuguous temporal regions where a quantity is between 
	two values, preventing false transitions that are too brief
	
	Args:
		arr: length-N numpy array holding data
		fps: float; frames per second
		traj_ranges: list of integer ranges corresponding to each distinct
			trajectory. If None, then array is a single trajectory
		on_thresh: 2-element list; lower and upper bound of vals that 
			define a "hit" region. 
		off_thresh: 2-element list; lower and upper bound of vals that 
			define the end of a hit. The lower/upper bound of this list does not 
			have to coincide with the upper/lower bound of on_thresh.
		min_on_dur: float; time of minimum length of hit to be valid 
		min_off_dur: float; time of minimum length of non-hit to count as 
			a transition out of the hit region
	
	Returns:
		bin_hits: length-N numpy binary array of hit times
		hits: (N, 2) numpy array containing indices at which ons begin and end
	"""
	
	# Get locations of below and above threshold points; ignore Nan warnings
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		within_on = 1.*((arr >= on_thresh[0]) & (arr <= on_thresh[1]))
		within_off = 1.*((arr >= off_thresh[0]) & (arr <= off_thresh[1]))
		
	on_begs = np.where(np.diff(within_on) == 1)[0]
	on_ends = np.where(np.diff(within_on) == -1)[0] 
	off_begs = np.where(np.diff(within_off) == 1)[0]
	off_ends = np.where(np.diff(within_off) == -1)[0] 
	
	if traj_ranges is None:
		traj_ranges = [np.arange(len(arr))]
	
	# Hold all legitimate stop starts and ends; these are the main structures.
	# The indices saved here will be those of the full exp_matrix_dict
	masked_hit_begs = []
	masked_hit_ends = []
	bin_hits = np.zeros(arr.shape)
	
	for iR, iRange in enumerate(traj_ranges):
		
		min_on_frms = int(min_on_dur*fps)
		min_off_frms = int(min_off_dur*fps)
		
		# These are nominal begs and ends of on-regions
		on_begs_in_rng = np.intersect1d(on_begs, iRange)
		on_ends_in_rng = np.intersect1d(on_ends, iRange)
		
		# Line up the begs and ends of on regions; add endpoints
		if len(on_begs_in_rng)*len(on_ends_in_rng) == 0:
			continue
		if on_ends_in_rng[0] < on_begs_in_rng[0]:
			on_begs_in_rng = np.hstack((iRange[0], on_begs_in_rng))
		if len(on_begs_in_rng) == len(on_ends_in_rng) + 1:
			on_ends_in_rng = np.hstack((on_ends_in_rng, iRange[-1]))
		
		# Add region to events if not too short
		for iB in range(len(on_begs_in_rng)):
			on_rng = np.arange(on_begs_in_rng[iB], on_ends_in_rng[iB])
			if len(on_rng) > min_on_frms:
				bin_hits[on_rng] = 1
		
		# Remove all off-regions that are too short
		on_bin = 1.*(bin_hits[iRange] > 0)
		off_begs = np.where(np.diff(on_bin) == -1)[0]
		off_ends = np.where(np.diff(on_bin) == 1)[0]
		if on_bin[0] == 0:
			off_begs = np.hstack(([0], off_begs))
		if on_bin[-1] == 0:
			off_ends = np.hstack((off_ends, [len(iRange) - 1]))
		for iB in range(len(off_begs)):
			off_rng = np.arange(off_begs[iB], off_ends[iB] + 1) + iRange[0]
			if len(off_rng) < min_off_frms:
				bin_hits[off_rng] = 1
		
		# Below is for binary-hits array; include endpoints
		# Organize on-off regions to remove ones that overlap boundaries
		true_on_begs = np.where(np.diff(bin_hits[iRange]) == 1)[0]
		true_on_ends = np.where(np.diff(bin_hits[iRange]) == -1)[0]
		
		# If either overlaps the boundary
		if len(true_on_ends)*len(true_on_begs) == 0:
			continue
		if true_on_ends[0] < true_on_begs[0]:
			true_on_begs = np.hstack(([0], true_on_begs))
		if bin_hits[iRange][-1] == 1:
			true_on_ends = np.hstack((true_on_ends, [len(iRange) - 1]))
		
		# Save to array; increase by 1 at each on-region
		for iB in range(len(true_on_begs)):
			on_rng = np.arange(true_on_begs[iB], true_on_ends[iB] + 1) + iRange[0]
			bin_hits[on_rng] = iB + 1
		
		# Below is for defining ends of ranges.
		# Do not count stops at boundaries
		if bin_hits[iRange[0]] > 0:
			true_on_begs = true_on_begs[1:]
			true_on_ends = true_on_ends[1:]
		if bin_hits[iRange[-1]] > 0:
			true_on_begs = true_on_begs[:-1]
			true_on_ends = true_on_ends[:-1]
		
		masked_hit_begs.extend(true_on_begs + iRange[0])
		masked_hit_ends.extend(true_on_ends + iRange[0])
	
	# Save the endpoints of all ranges
	hits = np.vstack((np.array(masked_hit_begs), np.array(masked_hit_ends))).T
	
	return bin_hits, hits


class gen_stops_turns(object):
	"""
	Class for generating stops and walk times for experimental data.
	"""
	
	def __init__(self, subdir='2019-8-3'):
		
		self.exp_dirs = glob("%s/%s/*/" % (get_data_dir(), subdir))
		
	def gen_st_events_all_exp_dirs(self, st_threshs=[0, 2], wk_threshs=[2, 100], 
								   min_st_dur=0.3, min_wk_dur=0.3):
		"""
		Generate stop events for all experiments in subdir
		"""
		
		for exp_dir in self.exp_dirs:
			self.gen_st_events_single_exp_dir(exp_dir, st_threshs, wk_threshs, 
											  min_st_dur, min_wk_dur)
				
	def gen_tn_events_all_exp_dirs(self, tn_threshs=[150, 2000], 
								   st_threshs=[0, 150], min_tn_dur=0.1, 
								   min_st_dur=0.1):
		"""
		Generate turn events for all experiments in subdir
		"""
		
		for exp_dir in self.exp_dirs:
			self.gen_tn_events_single_exp_dir(exp_dir, tn_threshs, st_threshs, 
											  min_tn_dur, min_st_dur)

	def gen_st_events_single_exp_dir(self, exp_dir, st_threshs=[0, 2], 
									 wk_threshs=[2, 100], min_st_dur=0.3, 
									 min_wk_dur=0.3):
		"""
		Generate stop events for single experiment, exp_dir
		"""
		
		obj, _, _ = load_exp_data(exp_dir)
		exp_matrix = load_exp_matrix(obj)
		metadata = load_metadata(obj)
		
		spd = exp_matrix['spd_smooth']
		fps = obj.rec_rate
		traj_ranges = get_traj_ranges(exp_matrix)
	
		bin_hits, stop_idxs = calc_thresholded_events(arr=spd, fps=fps,
			traj_ranges=traj_ranges, on_thresh=st_threshs, 
			off_thresh=wk_threshs, min_on_dur=min_st_dur, 
			min_off_dur=min_wk_dur)
		
		exp_matrix['stops'] = dict()
		exp_matrix['stops']['bin'] = bin_hits
		exp_matrix['stops']['idxs'] = stop_idxs
		
		metadata['stops'] = dict()
		metadata['stops']['st_threshs'] = st_threshs
		metadata['stops']['wk_threshs'] = wk_threshs
		metadata['stops']['min_st_dur'] = min_st_dur
		metadata['stops']['min_wk_dur'] = min_wk_dur

		save_exp_matrix(exp_matrix, obj)
		save_metadata(metadata, obj)
		
	def gen_tn_events_single_exp_dir(self, exp_dir, tn_threshs=[150, 2000], 
									 st_threshs=[0, 150], min_tn_dur=0.1, 
									 min_st_dur=0.1):
		"""
		Generate turn events for single experiment, exp_dir
		
		tn_threshs: 2-element list 
			Range of dtheta/dt in deg/s which counts as a turn event
		st_threshs: 2-element list 
			Range of dtheta/dt in deg/s which counts as a straight walk
		min_tn_dur: float
			Minimum duration of a turn in seconds
		min_st_dur: float
			Minimum duration of a straight walk in seconds
		"""
		
		obj, _, _ = load_exp_data(exp_dir)
		exp_matrix = load_exp_matrix(obj)
		metadata = load_metadata(obj)
		
		omegas = abs(exp_matrix['dtheta_smooth'])
		fps = obj.rec_rate
		traj_ranges = get_traj_ranges(exp_matrix)
	
		bin_hits, turn_idxs = calc_thresholded_events(arr=omegas, fps=fps,
			traj_ranges=traj_ranges, on_thresh=tn_threshs, 
			off_thresh=st_threshs, min_on_dur=min_tn_dur, 
			min_off_dur=min_st_dur)
		
		exp_matrix['turns'] = dict()
		exp_matrix['turns']['bin'] = bin_hits
		exp_matrix['turns']['idxs'] = turn_idxs
		
		metadata['turns'] = dict()
		metadata['turns']['tn_threshs'] = tn_threshs
		metadata['turns']['st_threshs'] = st_threshs
		metadata['turns']['min_tn_dur'] = min_tn_dur
		metadata['turns']['min_st_dur'] = min_st_dur

		save_exp_matrix(exp_matrix, obj)
		save_metadata(metadata, obj)
