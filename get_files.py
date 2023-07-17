from torch import load
from numpy import linspace, zeros, log10
import os
import shutil

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Put the files in folder')


    #folder to create
    folder_to_create = 'npy_files'

    #go to the folder trained_models, and copy all the files to the folder_to_create folder
    for folder in os.listdir('trained_models'):
        #skip if the folder does not contain the word RC
        if 'RC' not in folder:
            continue
        print('Retrieving files from folder', folder)
        path = os.path.join('trained_models', folder)
        #go to prune/latest_exp folder
        path = os.path.join(path, 'prune', 'latest_exp')
        #get the 'epochs_data.npy' file
        file = os.path.join(path, 'epochs_data.npy')

        #copy the file to the folder_to_create folder
        shutil.copy(file, folder_to_create)

        #same with setup.log
        file = os.path.join(path, 'setup.log')
        shutil.copy(file, folder_to_create)

print('Done!')

if __name__ == '__main__':
    main()