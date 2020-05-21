import numpy as np
from PIL import Image, ImageChops
from glob import glob
import matplotlib.pyplot as plt
from skimage import color
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from skimage.transform import resize
from sklearn.tree import DecisionTreeClassifier


class DogClassifier:
    
    # Read in and prepare data
    def __init__(self): 
        
        files = glob('./images/*')
        data, target = [], []
        
        for file in files:
            
            raw = Image.open(file)
            
            # Trim whitespace around image
            bg = Image.new(raw.mode, raw.size, raw.getpixel((0,0)))
            diff = ImageChops.difference(raw, bg)
            diff = ImageChops.add(diff, diff, 2.0, -100)
            img =  np.array(raw.crop(diff.getbbox()))
            
            # Uniform dimensionality
            img = resize(img, (500,500), anti_aliasing=True)
            img = color.colorconv.rgb2gray(img)
            
            # 0=borzoi 1=dachshund
            if 'borzoi' in file:
                data.append(np.concatenate([i for i in img]))
                target.append(0)
            if 'dachshund' in file:
                data.append(np.concatenate([i for i in img]))
                target.append(1)
                
            if 'test' in file:
                self.t = img
            
        self.X = np.array(data)
        self.y = np.array(target)
    
    # Train the classifier. Only fit the classifier if arg fit==True, otherwise 
    # training data _X_train and _y_train is not needed
    def train(self, classif, _X_train=None, _y_train=None, fit=True):
        
        self.clf = classif
        if fit==True:
            self.clf.fit(_X_train, _y_train)
        return self.clf
        
    # Make estimate given testing data
    def predict(self, _X_test):
        
        self.predictions = self.clf.predict(_X_test)
        return self.predictions
   
    # Evaluate the classifier's performance
    def eval(self, _y_test):
        
        self.score = np.mean(self.predictions==_y_test)
        return self.score
    
    # Test the classifier with new images
    def deploy(self):
        
        image = np.array([np.concatenate([i for i in self.t])])
        pred = self.clf.predict(image)
        [print('DACHSHUND') if pred == 1 else print('BORZOI')]
        
        try:
            prob = self.clf.predict_proba(image)[0][pred][0]
            print('@ {}%'.format(format(prob*100, '3.1f')))
        except AttributeError:
            pass
        
        plt.figure()
        plt.imshow(self.t)


def grid_search(): 
    """Grid search parameter optimization using k-fold cross validation"""
    
    brain = DogClassifier()    
    X_train, X_test, y_train, y_test = train_test_split(brain.X, brain.y)
    
    classifiers = {'SGD': SGDClassifier(),
                   'SVC': SVC(gamma='scale'),
                   'KNN': KNeighborsClassifier(),
                   'DTC': DecisionTreeClassifier()}
    
    parameters = {'DTC': {'clf__criterion': ('entropy', 'gini'),
                          'clf__splitter': ('best', 'random'),
                          'clf__max_features': ('auto', 'sqrt', 'log2')},
                  'SGD': {'clf__penalty': ('l1', 'l2', 'elasticnet'),
                          'clf__alpha': (5e-5, 1e-4, 5e-4),
                          'clf__loss': ('hinge', 'log', 'squared_hinge', 'perceptron')},
                  'SVC': {'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                          'clf__degree': (2, 3, 4, 5), 
                          'clf__C': (0.1, 0.5, 1.0, 2.0)},
                  'KNN': {'clf__algorithm': ('ball_tree', 'auto', 'kd_tree', 'brute'),
                          'clf__weights': ('uniform', 'distance'),
                          'clf__n_neighbors': (5, 10, 15, 20)} }
                  
    for name, params in parameters.items():
        
        brain.train(classifiers[name], fit=False)
        gs_clf = GridSearchCV(brain.clf, params, cv=2, refit=True)
        gs_clf.fit(X_train, y_train)
        
        print(name, gs_clf.best_score_, gs_clf.best_params_, '\n')  
        
        
def cross_validation(): 
    """Stratified k-fold cross validation using splits of 5. Reports accuracies and 
    F1 scores for given classifiers using splits of 5"""
    
    brain = DogClassifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # Parameters optimized by grid search
    classifiers = {'SGD': SGDClassifier(penalty='l1', alpha=5e-5, loss='log'),
                   'SVC': SVC(gamma='scale', kernel='poly', degree=2, C=0.5),
                   'KNN': KNeighborsClassifier(algorithm='ball_tree', weights='distance', n_neighbors=15),
                   'DTC': DecisionTreeClassifier(criterion='entropy', splitter='random', max_features='auto')}

    for train_index, test_index in skf.split(brain.X, brain.y):
        X_train, X_test = brain.X[train_index], brain.X[test_index]
        y_train, y_test = brain.y[train_index], brain.y[test_index]

        for name, classif in classifiers.items():
            brain.train(classif, X_train, y_train)
            preds = brain.predict(X_test)
            f1 = f1_score(y_test, preds)
            acc = brain.eval(y_test)
            print(name, format(acc, '1.3f'), format(f1, '1.3f'))
        print()


def one_time():
    """One-time test of classifier performance. Returns only accuracy"""
    
    brain = DogClassifier()
    X_train, X_test, y_train, y_test = train_test_split(brain.X, brain.y)
    
    clf = SGDClassifier() # parameters optional
    brain.train(clf, X_train, y_train)
    brain.predict(X_test)
    
    print(brain.eval(y_test))
    
    
def deployment_test():
    """Deploy the classifier on a new image"""

    brain = DogClassifier()
    classifier = SGDClassifier(penalty='l1', alpha=5e-5, loss='log', random_state=37)
    brain.train(classifier, brain.X, brain.y)
    brain.deploy()
    
    
if __name__ == '__main__':
    
    # one_time()
    # cross_validation()
    # grid_search()
    deployment_test()
    
    pass
 

    
    