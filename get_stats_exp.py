import argparse
import matplotlib.pyplot as plt
from torch import load
from numpy import linspace, zeros, load
import os

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Do plots on the evolution of some metrics through pruning.')

    # Add command line arguments
    parser.add_argument('--exp-name', type=str, help='Name of the folder (and experiment) to look into.')
    #parser.add_argument('--n', type=int, help='Number of points to plot', default=1000)
    #parser.add_argument('--save-path', type=str, help='Path to the image of the plot')


    # Parse the command line arguments
    args = parser.parse_args()

    # Access the argument values
    folder = args.exp_name
    #n_points = args.n

    #go to the folder 'latest_exp' inside folder
    folder = os.path.join(folder, 'latest_exp')

    #open the file 'epochs_data.npy' inside folder
    file_path = os.path.join(folder, 'epochs_data.npy')

    l = load(file_path, allow_pickle=True)

    print(l)