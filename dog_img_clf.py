import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import sys
sys.path.append('path to Window_Filter.py')
from window_filter import window



class DogClassifier:
    
    # Read in and prepare data. Accepts name of folder containing training data
    # and square dimension size of the data as arguments
    def __init__(self, folder, size=200): 
        
        self.s = size
        files = glob(f'./{folder}/*')
        data, target = [], []

        # Iterate over every file in images folder
        for file in files:
            
            # Read image and preprocess
            img = self.prep(file)
            
            # 0=borzoi, 1=dachshund
            if 'borzoi' in file:
                data.append(img)
                target.append(0)
            if 'dachshund' in file:
                data.append(img)
                target.append(1)   
            
        self.X = np.array(data)
        self.y = np.array(target)
    
    # Preprocess image data
    def prep(self, f):
        
        img_raw = Image.open(f).resize(
            (self.s, self.s)
            )
        
        return window(img_raw)
            
    # Train the classifier. Only fit the classifier if arg fit==True, otherwise 
    # training data X_train and y_train is not needed
    def train(self, classif, X_train=None, y_train=None, fit=True):
        
        self.clf = Pipeline([
            ('clf', classif),
            ])
        
        if fit:
            self.clf.fit(X_train, y_train)
        
        return self.clf
        
    # Make estimate given testing data
    def predict(self, X_test):
        
        self.predictions = self.clf.predict(X_test)
        return self.predictions
   
    # Evaluate the classifier's performance
    def evaluate(self, y_test):
        
        # Return accuracy score
        self.score = np.mean(self.predictions==y_test)
        return self.score



def grid_search(brain): 
    """Grid search parameter optimization using k-fold cross validation"""
    
    f = open('params.txt', 'w')
    
    classifiers = {
        'SVC'      : svm.SVC(random_state=42),
        'NuSVC'    : svm.NuSVC(random_state=42),
        'LinearSVC': svm.LinearSVC(random_state=42),
        'SGD'      : SGDClassifier(random_state=42, loss='hinge'),
        }
    
    parameters = {
        'SVC': {
            'clf__kernel'       : ('poly', 'rbf', 'sigmoid'),
            'clf__degree'       : (2, 3, 4, 5), 
            'clf__gamma'        : ('scale', 'auto'),
            'clf__tol'          : (1e-2, 1e-3, 1e-4),
            },
        'NuSVC': {
            'clf__nu'           : (.4, .5, .6),
            'clf__kernel'       : ('poly', 'rbf', 'sigmoid'),
            'clf__degree'       : (2, 3, 4, 5), 
            'clf__gamma'        : ('scale', 'auto', 0.1),
            'clf__tol'          : (1e-2, 1e-3, 1e-4),
            },
        'LinearSVC': {      
            'clf__penalty'      : ('l1', 'l2'),
            'clf__loss'         : ('hinge', 'squared_hinge'),
            'clf__C'            : (1, 0.1, 0.01, 0.001),
            'clf__tol'          : (1e-2, 1e-3, 1e-4),
            },
        'SGD': {
            'clf__penalty'      : ('l1', 'l2', 'elasticnet'),
            'clf__alpha'        : (0.01, 0.001, 0.0001),
            'clf__tol'          : (1e-2, 1e-3, 1e-4),
            'clf__learning_rate': ('constant', 'optimal', 'invscaling'),
            'clf__eta0'         : (0.5, 0.1, 0.001),
            },
        }
                  
    for name, params in parameters.items():
        
        brain.train(classifiers[name], fit=False)
        gs_clf = GridSearchCV(brain.clf, params, cv=3, refit=True, scoring='f1')
        gs_clf.fit(brain.X, brain.y)
        
        print(name, gs_clf.best_score_, gs_clf.best_params_, '\n', file=f) 
    
    f.close()
        
        
        
def cross_validation(brain): 
    """Stratified k-fold cross validation using splits of 5. Reports accuracies and 
    F1 scores for given classifiers using splits of 7"""
    
    skf = StratifiedKFold(n_splits=7, shuffle=True)
    cm = []
    
    name = 'NuSVC'
    classifiers = { 
        'SVC'      : svm.SVC(degree=2, gamma='scale', kernel='linear', tol=0.01),
        'NuSVC'    : svm.NuSVC(degree=2, gamma='scale', kernel='rbf', nu=0.5, tol=0.01),
        'LinearSVC': svm.LinearSVC(C=0.01, loss='squared_hinge', penalty='l2', tol=0.01),
        'SGD'      : SGDClassifier(loss='hinge', alpha=0.001, eta0=0.001, learning_rate='invscaling', penalty='l2', tol=0.0001),
        }
            
    for train_index, test_index in skf.split(brain.X, brain.y):
        
        X_train, X_test = brain.X[train_index], brain.X[test_index]
        y_train, y_test = brain.y[train_index], brain.y[test_index]
        
        classifier = classifiers[name]
        brain.train(classifier, X_train, y_train)
        preds = brain.predict(X_test)
        
        f1 = f1_score(y_test, preds)
        acc = brain.evaluate(y_test)
        cm.append(confusion_matrix(y_test, preds))
        print(name, format(acc, '1.3f'), format(f1, '1.3f'), '\n')
    
    plt.imshow(sum(cm), interpolation='nearest')
    plt.title(name)
    plt.colorbar()
    plt.xticks(np.arange(2))
    plt.yticks(np.arange(2))    
    plt.ylabel('true value')
    plt.xlabel('prediction')
    
    

def build_pkl(brain):
    
    with open('model.pkl', 'wb') as f:
        
        clf = svm.NuSVC(kernel='rbf', degree=2, gamma='scale', nu=0.5, tol=0.01)
        brain.train(clf, brain.X, brain.y)
        
        pickle.dump([brain.prep, brain.clf], f)
    
    

def deploy_pkl(file):
    
    with open('model.pkl', 'rb') as f:

        model = pickle.load(f)
        
        img = np.array([model[0](file)])
        pred = model[1].predict(img)
        
        [print('DACHSHUND') if pred == 1 else print('BORZOI')]
    
    
    
if __name__ == '__main__':
        
    model = DogClassifier(folder='images', size=200)
    
    # grid_search(model)
    # cross_validation(model)
    # build_pkl(model)
    
    # file = Image.open('images\\test.jpg')
    # deploy_pkl(model, file)
    
