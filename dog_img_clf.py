import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from skimage import color
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



class DogClassifier:
    
    # Read in and prepare data
    def __init__(self): 
        
        files = glob('./images/*')
        data, target = [], []

        # Iterate over every file in images folder
        for file in files:
            
            raw = Image.open(file)
            
            # Uniform dimensionality
            img = np.array(raw)
            img = resize(img, (500,500), anti_aliasing=True)
            img = color.colorconv.rgb2gray(img)
            
            # Borzoi: 0, Dachshund: 1
            if 'borzoi' in file:
                data.append(np.concatenate([i for i in img]))
                target.append(0)
            if 'dachshund' in file:
                data.append(np.concatenate([i for i in img]))
                target.append(1)            
            
        self.X = np.array(data)
        self.y = np.array(target)
    
    # Train the classifier. Only fit the classifier if arg fit==True, otherwise 
    # training data _X_train and _y_train is not needed
    def train(self, classif, X_train=None, y_train=None, fit=True):
        
        self.clf = Pipeline(
            [('clf', classif)],
            )
        
        if fit==True:
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
    
    # Test the classifier with new images
    def deploy(self, raw):
        
        # Preprocess image in same way as training data
        img = np.array(raw)
        img = resize(img, (500,500), anti_aliasing=True)
        img = color.colorconv.rgb2gray(img)
        
        self.test = np.array([np.concatenate([i for i in img])])
        pred = self.clf.predict(self.test)
        [print('DACHSHUND') if pred == 1 else print('BORZOI')]
        
        # Print the probability of accuracy
        try:
            prob = self.clf.predict_proba(self.test)[0][pred][0]
            print('@ {}%'.format(format(prob*100, '3.1f')))
        except AttributeError:
            pass
        
        # Display the computer vision image
        plt.figure()
        plt.imshow(img)



def grid_search(brain): 
    """Grid search parameter optimization using k-fold cross validation"""
    
    f = open('params.txt', 'a')
    
    classifiers = {
        'SVC': SVC(gamma='scale'),
        'SGD': SGDClassifier(),
        'KNN': KNeighborsClassifier(),
        'LDA': LinearDiscriminantAnalysis(),
        'GNB' : GaussianNB(),
        }
    
    parameters = {
        'SVC': {
            'clf__kernel'     : ('linear', 'poly', 'sigmoid'),
            'clf__degree'     : (2, 3, 4), 
            'clf__C'          : (1, 0.1, 0.01, 0.001),
            },
        'SGD': {
            'clf__penalty'    : ('l1', 'l2', 'elasticnet'),
            'clf__alpha'      : (0.01, 0.001, 0.0001, 0.00001),
            'clf__loss'       : ('hinge', 'log'),
            },
        'KNN': {
            'clf__algorithm'  : ('ball_tree', 'auto', 'kd_tree'),
            'clf__weights'    : ('uniform', 'distance'),
            'clf__n_neighbors': (5, 10, 15),
            },
        'LDA': {
            'clf__solver'     : ('svd', 'lsqr', 'eigen'),
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
    F1 scores for given classifiers using splits of 5"""
    
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    name = 'KNN'

    classifiers = {
        'SVC': SVC(gamma='scale', C=1, kernel='poly', degree=3),
        'SGD': SGDClassifier(alpha=0.0001, loss='hinge', penalty='elasticnet'),
        'KNN': KNeighborsClassifier(algorithm='ball_tree', n_neighbors=15, weights='uniform'),
        'LDA': LinearDiscriminantAnalysis(solver='svd'),
        'GNB' : GaussianNB(),
        }
    
    for train_index, test_index in skf.split(brain.X, brain.y):
        
        X_train, X_test = brain.X[train_index], brain.X[test_index]
        y_train, y_test = brain.y[train_index], brain.y[test_index]
        
        classifier = classifiers[name]
        brain.train(classifier, X_train, y_train)
        preds = brain.predict(X_test)
        
        f1 = f1_score(y_test, preds)
        acc = brain.evaluate(y_test)
        print(name, format(acc, '1.3f'), format(f1, '1.3f'), '\n')
    
    
    
def deployment_test(brain):
    """Deploy the classifier on a new image"""
    
    path = os.getcwd()+'\\images'
    image = Image.open(os.path.join(path, 'test.jpg'))
    
    classifier = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=15, weights='uniform')
    brain.train(classifier, brain.X, brain.y)
    brain.deploy(image)
    
   
    
if __name__ == '__main__':
    
    model = DogClassifier()
    
#    grid_search(model)
#    cross_validation(model)
#    deployment_test(model)
    
