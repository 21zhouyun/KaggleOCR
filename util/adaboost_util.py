import preprocess
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
 
def cross_validation(X, y):
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    assert(len(y) == len(X))
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
 
    depth = [8, 16, 32, 64]
    split = [1, 2, 4, 8, 16, 32, 64]
    best_score = 0 
    best_train_score = 0
    best_param = None
    for d in depth:
        for s in split:
            estimator = DecisionTreeClassifier(max_features='sqrt', max_depth = d, min_samples_split = s)
            model = AdaBoostClassifier(n_estimators=500, base_estimator = estimator)
            model = model.fit(X_train, y_train)
            print "Depth: %d  split: %d" % (d, s)
            print "Model trainning score:"
            score_train = model.score(X_train, y_train)
            print score_train
            #ax.scatter(d, s, score_train, c='b', marker='o')
            print "Model test score:"
            score_test = model.score(X_test, y_test)
            print score_test
            #ax.scatter(d, s, score_test, c='r', marker='^')
 
            if score_test > best_score:
                best_score = score_test
                best_train_score = score_train
                best_param = model.get_params()
    print "=================="
    print best_train_score
    print best_score
    print best_param
    return best_param
 
def train_random_forest(X, y, mss, md):
    model = RandomForestClassifier(n_estimators=1280, criterion="entropy", max_features="sqrt", max_depth=md, min_samples_split=mss, n_jobs=-1)
    model = model.fit(X, y)
    score = model.score(X, y)
    print "Model Trainning Score: %s" % score
    return model

def train_random_forest_with_params(X, y, params):
    model = RandomForestClassifier()
    model.set_params(params)
    model = model.fit(X, y)
    score = model.score(X, y)
    print "Model Trainning Score: %s" % score
    return model