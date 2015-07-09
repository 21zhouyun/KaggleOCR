from abstract_model import AbstractModel
from util import svm_util
from util import util
from sklearn.decomposition import PCA

class PCA_SVM(AbstractModel):

	def train(self, train_X, train_y, params=None):
		print "Running PCA"
		self.pca = PCA(n_components=36, whiten=True)
		self.pca.fit(train_X)
		train_X = self.pca.transform(train_X)

		if not params:
			print "Running cross validation"
			params = svm_util.cross_validation(train_X, train_y, "rbf")
			print "Training model with best params"
		self.model = svm_util.train_svc(train_X, train_y, params["kernel"], params[
			"C"], params["gamma"])
		print "Finished building model"

		return self.model

	def predict(self, test_X):
		print "Using the PCA model to transform the test data"
		test_X = self.pca.transform(test_X)
		print "Predicting"
		self.predict = self.model.predict(test_X)

		return self.predict
