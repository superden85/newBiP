import os
import shutil

def main():

    #folder to create
    folder_to_create = 'npy_files'

    #create the folder if not exists
    if not os.path.exists(folder_to_create):
        os.mkdir(folder_to_create)
    

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
        destination_path = os.path.join(folder_to_create, folder)
        destination_path = os.path.join(destination_path, 'epochs_data.npy')
        #copy the file to the folder_to_create folder
        shutil.copy(file, destination_path)

        #same with setup.log
        file = os.path.join(path, 'setup.log')
        destination_path = os.path.join(folder_to_create, folder)
        destination_path = os.path.join(destination_path, 'setup.log')

        #copy the file to the folder_to_create folder
        shutil.copy(file, destination_path)

    print('Done!')

if __name__ == '__main__':
    main()