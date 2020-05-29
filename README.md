# Image-Classifier 
*work in progress*

This project is a binary classifier for images of two different kinds of dogs. Several support vector machines are optimized and tested to find the most accurate classification algorithm. Sci-Kit Learn is the machine learning library used in building the model.

The file dog_img_clf.py is the script for a basic supervised learning image classification model. The data are excluded from this repository but can be provided on request or supplemented with one's own image wrangling. The dataset is structured as a folder of roughly 500 .jpg files. The files are titled according to an index number and the breed of dog that they represent ("borzoi" or "dachshund"). The file img_save.py pulls images from a folder of Google Images bulk downloads and saves them into the "images" folder.

The script creates the classifier object ("DogClassifier"), which reads in the images and formats the data appropriately. The data are preprocessed by formatting a uniform dimension then using image thresholding to isolate the most significant features. Other preprocessing techniques, such as mean-centering/standardization and principal components analysis, were tested in conjunction with image thresholding, but were shown to diminishing the accuracy of the model.

The object also provides attributes for training, predicting, and evaluating (based on accuracy) the classifier. Grid search cross validation was used to optimize the hyper-parameters for each classifier tested. Stratified k-fold cross validation was then used to identify the best classifier given these hyper-parameters. The best classifier was then initiated, trained, and pickled to be used for deployment. The script tests a SVC (Support Vector Classifier), a Nu SVC, a Linear SVC, and a Stochastic Gradient Descent SVC. 
