"""
Functions for controlling input/output through a DAQ. 

Created by Nirag Kadakia at 13:39 01-13-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

class daq():
	"""
	Example for a generic daq class. Currently include methods for setting
	MFC voltages and valve states.
	"""
	
	def __init(self):
		"""
		## TODO docstrings
		## TODO Automatically find weakrefs and delete in saving data?
		"""
		
		# Device object
		self.dev = None
		
		# Holds the port names, addresses, and descriptions for ports 
		# controlling the MFCs on the daq
		self.MFC_address_names = ['DAC0', 'DAC1']
		self.MFC_addresses = [5000, 5002]
		self.MFC_descriptors = ['description of MFC 1', 
								'description of MFC 2']
		
		# Minimum and maximum voltages for each MFC
		self.MFC_min_Vs = [0, 0]
		self.MFC_max_Vs = [5, 5]
		
		# Holds the port names, addresses, and descriptions for ports 
		# controlling the valves on the daq
		self.valve_address_names = ['FIO4', 'FIO5']
		self.valve_addresses = [6004, 6005]
		self.valve_descriptors = ['description of valve 1', 
								  'description of valve 2']
		
	def set_MFC(self, num, V):
		"""
		Analog voltage for MFC numbered `num'
		"""
		
		if V > self.MFC_max_Vs[num]:
			print('MFC %d max voltage is 5. Setting to 5.' % num)
			V = self.MFC_max_Vs[num]
		if V < self.MFC_min_Vs[num]:
			print('MFC %d min voltage is 0. Setting to 0.' % num)
			V = self.MFC_min_Vs[num]
		
		self.MFCs[num] = V
		self.dev.writeRegister(self.MFC_addresses[num], self.MFCs[num])
							
	def set_valve(self, num, state):
		"""
		Digital state sets with 3.4V logic for valve. State 1 
		is open; 0 is closed.
		"""
		
		if state == 1:
			self.valves[num] = 1
		elif state == 0:
			self.valves[num] = 0
		else:
			print('Unknown state for valve. Setting to off')
			self.valves[num] = 0
			
		self.dev.writeRegister(self.valve_addresses[num], self.valves[num])
	

class daq_labjack_useries(daq):
	"""
	Class for setting voltages using a LabJack U3, U6 or U9 series.
	"""

	def __init__(self):
		"""
		Find daq and set addresses.
		"""
		
		import u3, u6, ue9
		
		self.dev = None
		devs = [u6.U6, u3.U3, ue9.UE9]
		dev_types = {6: "U6", 3: "U3", 9: "UE9"}
		for dev in devs:
			try:
				self.dev = dev()
				print("Found and opened a %s with serial # %s" %
					  (dev_types[self.dev.devType], self.dev.serialNumber))
				break
			except:
				pass
		
		if self.dev is None:
			print("Unable to find or open a LabJack daq. Quitting.")
			quit()
		
		if (dev_types[self.dev.devType] == 'U3') or \
		  (dev_types[self.dev.devType] == 'U6') or \
		  (dev_types[self.dev.devType] == 'UE9'):
			
			self.MFC_address_names = ['DAC0', 'DAC1']
			self.MFC_addresses = [5000, 5002]
			self.MFC_max_flow_rates = [1000.0, 200.0]
			self.MFC_descriptors = ['side jets 1000 mL/min', 
									'odor inlet 200 mL/min']
			self.MFC_min_Vs = [0, 0]
			self.MFC_max_Vs = [5, 5]
			
			self.valve_address_names = ['FIO4', 'FIO5']
			self.valve_addresses = [6004, 6005]
			self.valve_descriptors = ['side jets switch', 
									  'odor inlet ON or OFF']
			
			"""
			Other addresses not used
			AIN0: 0
			AIN1: 2
			AIN2: 4
			AIN4: 6
			FIO6: 6006
			FIO7: 6007
			"""
			
		else:
			print('Addresses for LabJack Type %s not set yet. Please find '\
					'online and add to script.' % dev_types[self.dev.devType])

		# Set all MFC voltages and valve states to 0 (OFF)
		self.MFCs = [0, 0]
		self.valves = [0, 0]
		
	def set_MFC(self, num, V):
		"""
		Analog voltage for MFC numbered `num'
		"""
		
		if V > self.MFC_max_Vs[num]:
			print('MFC %d max voltage is 5. Setting to 5.' % num)
			V = self.MFC_max_Vs[num]
		if V < self.MFC_min_Vs[num]:
			print('MFC %d min voltage is 0. Setting to 0.' % num)
			V = self.MFC_min_Vs[num]
		
		self.MFCs[num] = V
		self.dev.writeRegister(self.MFC_addresses[num], self.MFCs[num])
							
	def set_valve(self, num, state):
		"""
		Digital state sets with 3.4V logic for valve numbered `'num'. State 1
		is open; 0 is closed.
		"""
		
		if state == 1:
			self.valves[num] = 1
		elif state == 0:
			self.valves[num] = 0
		else:
			print('Unknown state for valve. Setting to off')
			self.valves[num] = 0
			
		self.dev.writeRegister(self.valve_addresses[num], self.valves[num])
