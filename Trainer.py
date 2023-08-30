import time
import ray
import math
import torch
from torch import nn

import numpy as np

from progress.bar import ChargingBar
from progress.spinner import PieSpinner

# Future trainer class
class Trainer():

	def __init__(self, replay_buffer, alpha_config):

		self.replay_buffer = replay_buffer
		self.alpha_config = alpha_config

		self.epochs_value_loss = []
		self.epochs_policy_loss = []
		self.epochs_combined_loss = []

		self.tests_value_loss = []
		self.tests_policy_loss = []
		self.tests_combined_loss = []

		self.train_global_value_loss = []
		self.train_global_policy_loss = []
		self.train_global_combined_loss = []

		self.test_global_value_loss = []
		self.test_global_policy_loss = []
		self.test_global_combined_loss = []

		return


	
	
	def reset_plots(self):
		
		self.epochs_value_loss = []
		self.epochs_policy_loss = []
		self.epochs_combined_loss = []

		self.tests_value_loss = []
		self.tests_policy_loss = []
		self.tests_combined_loss = []

		self.train_global_value_loss = []
		self.train_global_policy_loss = []
		self.train_global_combined_loss = []

		self.test_global_value_loss = []
		self.test_global_policy_loss = []
		self.test_global_combined_loss = []