import argparse
import matplotlib.pyplot as plt
from torch import load
from numpy import linspace, zeros
import os

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Superimpose the infos of models')

    # Add command line arguments
    parser.add_argument('--file', type=str, help='Text file with all the folder names')
    #parser.add_argument('--n', type=int, help='Number of points to plot', default=1000)
    #parser.add_argument('--save-path', type=str, help='Path to the image of the plot')


    # Parse the command line arguments
    args = parser.parse_args()

    # Access the argument values
    file = args.file

    folders = []
    labels = []

    # Open the text file for reading
    with open(file, 'r') as file:
        # Read the file line by line
        for line in file:
            # Split the line into two words
            folder, label = line.strip().split(' ')

            folders.append(folder)
            labels.append(label)

    
    #plot the l0 norm of each model
    for folder, label in zip(folders, labels):
        #the folder is inside 'trained_models/'
        folder = os.path.join('trained_models', folder)

        #we have to go inside the folder 'prune/latest_exp/'
        folder = os.path.join(folder, 'prune/latest_exp')

        #open the file 'epochs_data.npy' inside folder
        file_path = os.path.join(folder, 'epochs_data.npy')

        #print('searching in : ', file_path)

        l = load(file_path, allow_pickle=True)

        l0_list = []
        l1_list = []
        mini_list = []
        maxi_list = []

        treshold_list = []
        below_treshold_list = []

        for (l0, l1, mini, maxi, c) in l:
            l0_list.append(l0)
            l1_list.append(l1)
            mini_list.append(mini)
            maxi_list.append(maxi)
            
            s, t = c
            treshold_list = s
            below_treshold_list.append(t)
        
        #plot the l0 norm
        plt.plot(l0_list, label=label)
    

    plt.title('Evolution of the metrics through pruning')
    plt.xlabel('Iteration')
    plt.legend()
    # Save the plot
    save_directory = 'plots'
    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

    save_path = os.path.join(save_directory, 'plot.png')
    plt.savefig(save_path)

    print('Plot saved.')


if __name__ == '__main__':
    main()