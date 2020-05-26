from glob import glob
from PIL import Image
import os
    
path = os.getcwd()+'\\images'
files_B = glob('./downloads/borzoi dog - Google Search_files/*')
files_D = glob('./downloads/dachshund dog - Google Search_files/*')

count = 1

for file in files_B:

    try:
        img = Image.open(file)
        img.save(os.path.join(path, f'borzoi{count}.jpg'))
        count += 1
    
    except:
        pass
      
for file in files_D:

    try:
        img = Image.open(file)
        img.save(os.path.join(path, f'dachshund{count}.jpg'))
        count += 1
    
    except:
        pass
    
    
    
