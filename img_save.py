from glob import glob
from PIL import Image
import os
    
path = os.getcwd()+'\\images'
count = 1

files_B = glob('./downloads/borzoi dog - Google Search_files/*')
for file in files_B:

    try:
        img = Image.open(file)
        img.save(os.path.join(path, f'borzoi{count}.jpg'))
        count += 1
    except:
        pass
      
files_D = glob('./downloads/dachshund dog - Google Search_files/*')
for file in files_D:

    try:
        img = Image.open(file)
        img.save(os.path.join(path, f'dachshund{count}.jpg'))
        count += 1
    except:
        pass
    
    
    