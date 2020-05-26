# Image-Classifier 
*work in progress*

This project is a binary classifier for images of two different kinds of dogs. 

The file dog_img_clf.py is the script for a basic supervised learning image classification model. The data are excluded from this repository but can be provided on request or supplemented with one's own image wrangling. The dataset is structured as a folder of roughly 500 .jpg files. The files are titled according to an index number and the breed of dog that they represent ("borzoi" or "dachshund"). The file img_save.py pulls images from a folder of Google Images bulk downloads and saves them into the "images" folder.

The script creates the classifier object ("DogClassifier"), which reads in the images and formats the data appropriately. The object also provides attributes for training, predicting, evaluating, and deploying the classifier. Grid search cross validation was used to optimize the hyper-parameters for each classifier tested. Stratified k-fold cross validation was then used to select the best classifier given these hyper-parameters. The model can be deployed with the best classifier by including an image in the dataset with the title "test.jpg". At the moment, a K-Nearest Neighbors algorithm has been found to perform best.

Sci-Kit Learn is the machine learning library used in building the model.
