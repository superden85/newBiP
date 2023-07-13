import argparse
import matplotlib.pyplot as plt
from torch import load
from numpy import linspace, zeros, load
import os

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Do plots on the evolution of some metrics through pruning.')

    # Add command line arguments
    parser.add_argument('--folder', type=str, help='Name of the folder to look into.')
    #parser.add_argument('--n', type=int, help='Number of points to plot', default=1000)
    #parser.add_argument('--save-path', type=str, help='Path to the image of the plot')


    # Parse the command line arguments
    args = parser.parse_args()

    # Access the argument values
    folder = args.folder
    #n_points = args.n

    #the folder is inside 'trained_models/'
    folder = os.path.join('trained_models', folder)

    #we have to go inside the folder 'prune/latest_exp/'
    folder = os.path.join(folder, 'prune/latest_exp')

    #open the file 'epochs_data.npy' inside folder
    file_path = os.path.join(folder, 'epochs_data.npy')

    print('searching in : ', file_path)

    l = load(file_path, allow_pickle=True)

    print(len(l))
    print(l[0])

if __name__ == '__main__':
    main()