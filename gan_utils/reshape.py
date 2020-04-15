#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'gan-pikachu/reshape-labels'))
	print(os.getcwd())
except:
	pass

#%%
import os
import glob
from PIL import Image, ImageOps

input_dirname = os.path.join('..', 'dataset', 'downloads', '*', '*')
output_dirname = os.path.join('..', 'dataset', 'reshaped')
files = glob.glob(input_dirname)
reshaped_size = (256, 256)

for i, file in enumerate(files):
    index = i + 1
    try:
        image = Image.open(file)
    except IOError:
        continue
    reshaped = ImageOps.fit(image, reshaped_size, Image.NEAREST)
    converted = reshaped.convert('RGB')
    converted.save(os.path.join(output_dirname, f'{index}.jpg'))
    print(f'{index}: {file} was saved.')


