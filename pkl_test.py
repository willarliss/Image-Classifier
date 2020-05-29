import pickle
from PIL import Image
import numpy as np



def deploy(raw):

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        
        img = np.array([model[0](raw)])
        pred = model[1].predict(img)
        
        [print('DACHSHUND') if pred == 1 else print('BORZOI')]



while True:
    user = input('Image: ')
    img = Image.open(user)
    deploy(img)