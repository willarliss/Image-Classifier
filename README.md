# Image-Classifier
This project is a binary classifier for images of two different kinds of dogs.

This is the script for a basic supervised learning image classification model. The data and data wrangling files are excluded from this repository but can be provided on request or supplemented with one's own image wrangling. The dataset is structured as a folder of roughly 500 .jpg files. The files are titled according to an index number and the breed of dog that they represent ("borzoi" or "dachshund").

The script creates the classifier object ("DogClassifier"), which reads in the images and cleans the data appropriately then provides attributes for training, predicting, evaluating, and deploying the classifier. Grid search cross validation was used to optimize the hyper-parameters for each classifier tested. Stratified k-fold cross validation was then used to select the best classifier given these hyper-parameters. The model can be deployed with the best classifier by including an image in the dataset with the title "test.jpg".

Sci-Kit Learn is the machine learning library used in building the model.
