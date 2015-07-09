from util import util
import pickle

# AbstractModel is a wrapper of the sklearn model
class AbstractModel():
	def ___init__(self):
		self.model = None
		self.predict = None

	def train(self, train_X, train_y, params=None):
		raise NotImplementedError("subclass must implement train")

	def predict(self, test_X):
		raise NotImplementedError("subclass must implement predict")
