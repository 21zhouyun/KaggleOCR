import preprocess
from util import svm_util
from util import util
import random
from sklearn import svm, metrics
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pickle

from pca_svm_model import PCA_SVM

# Read in the PCA_SVM model trained on the raw data and then
# use it to generate the second data set

print "Getting the raw data"
train_X1, train_y1, test_X = preprocess.get_default_data()
# train_X200k, train_y200k = preprocess.get_200k_data()
# print "Spliting  the 200k data into 2 100k data set"
# train_X2 = train_X200k[0:100000,:]
# train_y2 = train_y200k[0:100000]
# train_X3 = train_X200k[100000:,:]
# train_y3 = train_y200k[100000:]

# print "Loading model1"
# model1 = pickle.load(open("models/PCA_SVM_raw.pkl", "rb"))
# print model1

# # Following the procedure described in 
# # improving performance in neural networks using a boosting algorithm
# print "Get classified and misclassified data"
# prediction = model.predict(train_X2)
# misclassified_index = util.get_misclassified_index(prediction, train_y2)
# classified_index = util.get_classified_index(prediction, train_y2)
# misclassified_X = train_X2[misclassified_index,:]
# misclassified_y = train_y2[misclassified_index]
# classified_X = train_X2[classified_index,:]
# classified_y = train_y2[classified_index]
# # Use half classified with half misclassified
# print "Generating new training data"
# random_index_misclassified = random.sample(range(len(train_X2)), len(misclassified_X))
# sample_X_misclassified = train_X2[random_index_misclassified,:]
# sample_y_misclassified = train_y2[random_index_misclassified]

# random_index_classified = random.sample(range(len(train_X2)), len(classified_X))
# sample_X_classified = train_X2[random_index_classified,:]
# sample_y_classified = train_y2[random_index_classified]

# new_train_X = np.vstack((sample_X_classified, sample_X_misclassified))
# new_train_y = np.hstack((sample_y_classified, sample_y_misclassified))

# # Write the new data set
# print "Writing new training data"
# util.write_label_pattern_pairs(new_train_y, new_train_X, "temp1.csv")



# Use the second data set to generate the second model

# train_X, train_y = preprocess.get_training_data("temp1.csv")
# model = PCA_SVM()
# model.train(train_X, train_y, params={'C':10, 'gamma':0.03,'kernel':'rbf'})
# print "dump temp1 model"
# pickle.dump(model, open("PCA_SVM_temp1.pkl", "wb"))

# print "Loading model2"
# model2 = pickle.load(open("models/PCA_SVM_temp1.pkl", "rb"))
# print model2

# # Generate the third dataset
# pred1 = model1.predict(train_X3)
# pred2 = model2.predict(train_X3)

# mis_match_index = util.get_misclassified_index(pred1, pred2)
# mis_match_X = train_X3[mis_match_index]
# mis_match_y = train_y3[mis_match_index]
# print "Writing new training data"
# util.write_label_pattern_pairs(mis_match_y, mis_match_X, "temp2.csv")

# train_X, train_y = preprocess.get_training_data("temp2.csv")
# model = PCA_SVM()
# model.train(train_X, train_y)
# print "dump temp2 model"
# pickle.dump(model, open("PCA_SVM_temp2.pkl", "wb"))

print "Loading model1"
model1 = pickle.load(open("models/PCA_SVM_raw.pkl", "rb"))
print model1
print "Loading model2"
model2 = pickle.load(open("models/PCA_SVM_temp1.pkl", "rb"))
print model2
print "Loading model3"
model3 = pickle.load(open("models/PCA_SVM_temp2.pkl", "rb"))
print model3

print "Predicting"
pred1 = model1.predict(test_X)
pred2 = model2.predict(test_X)
pred3 = model3.predict(test_X)

mis_match_index = util.get_misclassified_index(pred1, pred2)
pred = pred1
pred[np.array(range(len(pred1)))[mis_match_index]] = pred3[mis_match_index]
print "Writing result"
util.write_preds(pred, "Boosting_PCA_SVM.csv")