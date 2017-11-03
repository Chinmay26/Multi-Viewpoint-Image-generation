from abc import ABCMeta, abstractmethod

"""
	Abstract Base Class Model Definition
"""
class Model(metaclass=ABCMeta):
	"""
		Contains definition of how to pass input to the model
	"""
	@abstractmethod
	def get_feed_dict(self):
		pass

	"""
		This contains the actual juice - implementation of some Neural Net architecture
	"""
	@abstractmethod
	def build(self):
		pass