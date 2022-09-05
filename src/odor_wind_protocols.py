"""
Protocols for controlling MFCs and valves using a daq.

Created by Nirag Kadakia at 10:09 01-15-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
from daq_funcs import *


class odor_wind_protocol():
	"""
	General class for a generic odor/wind protocol. Include methods for 
	updating MFC and valve voltages and states in real time during 
	experiment.
	
	Attrs
	-------
	
	rec_rate: float
		Recording frame rate in fps
	daq: daq object
		Object to control the MFCs and valves
	"""
	
	def __init__(self, rec_rate, daq=daq_labjack_useries):
		"""
		Constructor class for modulating MFCs and valves for odor and wind.
		
		Args
		-------
	
		rec_rate: float
			Recording frame rate in fps
		daq: daq object
			Object to control the MFCs and valves
		"""
		
		
		self.rec_rate = rec_rate
		self.daq = daq()
		
	def update(self, frm_num):
		"""
		Update odor and wind stimulus at each frame.
		
		Args
		-------
		
		frm_num: int
			Video frame number.
		"""
				
		if np.mod(frm_num, self.mod) == 0:
			self.daq.set_MFC(0, 2.0)
			self.daq.set_valve(1, 1)
		else:
			self.daq.set_MFC(0, 0.0)
			self.daq.set_valve(1, 0)
	
	def terminate(self):
		"""
		"""
		
		self.daq.set_MFC(0, 0)
		self.daq.set_MFC(1, 0)
		self.daq.set_valve(0, 0)
		self.daq.set_valve(1, 0)
		

class Bernoulli_jets(odor_wind_protocol):
	"""
	Jets switch following a Bernoulli process.
	
	Attrs
	-------
	
	rec_rate: float
		Recording frame rate in fps
	daq: daq object
		Object to control the MFCs and valves
	jet_flow_rate: float
		Rate of jet MFC in mL/min
	jet_V: float
		MFC voltage corresponding to jet_flow_rate
	odor_flow_rate: float
		Rate of odor MFC in mL/min
	odor_V: float
		MFC voltage corresponding to odor_flow_rate
	odor_on_T: float
		Time at which to switch on the odor MFC to odor_flow_rate
	odor_on_frm: int
		Frame at which to switch on the odor MFC to odor_flow_rate
	jet_on_T: float
		Time at which to switch on the jet MFC to jet_flow_rate
	jet_on_frm: int
		Frame at which to switch on the jet MFC to jet_flow_rate
	seed: int
		Random number seed for repeatability
	switch_T: float
		Either L or R jet will be opened with equal probability at 
		intervals of switch_T. Thus, the interval lengths are 
		geometrically distributed with parameter p, with 
		mean switch_T/p = switch_T/0.5 = 2*switch_T
	switch_frms: int
		Number of frames corresponding to switch_T
	active_jet: list
		The active jet at each frame (0 or 1)
	"""
	
	def __init__(self, rec_rate, daq=daq_labjack_useries, jet_flow_rate=100,
				 odor_flow_rate=60, switch_T=0.1, odor_on_T=0, jet_on_T=0, 
				 seed=0):
		"""
		Constructor class for modulating MFCs and valves for odor and wind.
		
		Args
		-------
	
		rec_rate: float
			Recording frame rate in fps
		daq: daq object
			Object to control the MFCs and valves
		jet_flow_rate: float
			Rate of jet MFC in mL/min
		odor_flow_rate: float
			Rate of odor MFC in mL/min
		switch_T: float
			Either L or R jet will be opened with equal probability at 
			intervals of switch_T. Thus, the interval lengths are 
			geometrically distributed with parameter p, with 
			mean switch_T/p = switch_T/0.5 = 2*switch_T
		odor_on_T: float
			Time at which to switch on the odor MFC to odor_flow_rate
		jet_on_T: float
			Time at which to switch on the jet MFC to jet_flow_rate
		seed: int
			Random number seed for repeatability
		"""
		
		
		self.rec_rate = rec_rate
		self.daq = daq()
		
		assert jet_flow_rate <= self.daq.MFC_max_flow_rates[0], "Jet flow "\
			"rate set at %.2f, but maximum is %.2f mL/min" % (jet_flow_rate, 
			self.daq.MFC_max_flow_rates[0])
		self.jet_flow_rate = jet_flow_rate
		self.jet_V = jet_flow_rate/self.daq.MFC_max_flow_rates[0]*\
						self.daq.MFC_max_Vs[0]
		assert odor_flow_rate <= self.daq.MFC_max_flow_rates[1], "Odor flow "\
			"rate set at %.2f, but maximum is %.2f mL/min" % (odor_flow_rate, 
			self.daq.MFC_max_flow_rates[1])
		self.odor_flow_rate = odor_flow_rate
		self.odor_V = odor_flow_rate/self.daq.MFC_max_flow_rates[1]*\
						self.daq.MFC_max_Vs[1]
		
		# Open odor valve but set voltage to minimum
		self.daq.set_valve(1, 1)
		self.daq.set_MFC(1, self.daq.MFC_min_Vs[1])
		
		# Frames start at 1; this is minimum frame
		self.odor_on_T = odor_on_T
		self.odor_on_frm = int(self.odor_on_T*self.rec_rate) + 1
		self.jet_on_T = jet_on_T
		self.jet_on_frm = int(self.jet_on_T*self.rec_rate) + 1
		
		assert not ((switch_T == 0) and (jet_flow_rate > 0)), 'switch_T '\
			'cannot be zero if jet_flow_rate is nonzero.'
		self.switch_T = switch_T
		self.switch_frms = int(switch_T*rec_rate)
		self.active_jet = []
		
		np.random.seed(seed)
		
	def update(self, frm_num):
		"""
		At every switch_T, either L or R jet is opened with equal probability.
		
		Args
		-------
		
		frm_num: int
			Video frame number.
		"""
		
		# Set jet and odor MFC to setpoint voltage after desired time
		if frm_num == self.odor_on_frm:
			self.daq.set_MFC(1, self.odor_V)
		if frm_num == self.jet_on_frm:
			self.daq.set_MFC(0, self.jet_V)
		
		if self.switch_frms > 0:
			if frm_num % self.switch_frms == 0:
				switch_choice = np.random.choice(2)
				self.daq.set_valve(0, switch_choice)
				self.active_jet.extend([switch_choice]*self.switch_frms)
		else:
			self.active_jet.append(np.nan)
			