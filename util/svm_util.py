from sklearn.svm import SVC
# validation
from sklearn import grid_search
import preprocess
 
 
def cross_validation(X, y, kernel):
    assert(len(y) == len(X))
 
    # Set the parameters by cross-validation
    svr = SVC(kernel=kernel)
    parameters = {"kernel":[kernel], "gamma":[0, 0.1, 0.03, 0.001], "C":[0.1, 1, 3, 10, 30, 100, 300]}
    clf = grid_search.GridSearchCV(svr, parameters, n_jobs=4, verbose=10, cv=3)
    clf.fit(X, y)
    print "best parameters"
    print clf.best_params_
    print "best score"
    print clf.best_score_
    return clf.best_params_
 

def train_svc(X, y, kernel, C, gamma):
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model = model.fit(X, y)
    print "SVC trainning accuracy: "
    print model.score(X, y)
    return model
