

class Node():

	def __init__(self, prior):
		self.visit_count = 0
		self.prior = prior
		self.value_sum = 0
		self.terminal_value = None
		self.children = {}
		self.to_play = -1
		
	def is_terminal(self):
		return self.terminal_value != None

	def expanded(self):
		return len(self.children) > 0

	def value(self):
		if self.visit_count == 0:
			return 0.0
		return self.value_sum / self.visit_count