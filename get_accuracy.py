import argparse
import matplotlib.pyplot as plt
from torch import load
from numpy import linspace, zeros, log10
import os
import re

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Plot the validation accuracies of the models')

    # Add command line arguments
    parser.add_argument('--file', type=str, help='Text file with all the paths to the setup logs')
    #parser.add_argument('--save-path', type=str, help='Path to the image of the plot')


    # Parse the command line arguments
    args = parser.parse_args()

    # Access the argument values
    file_path = args.file

    log_path_param = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Split the line into two words
            log_path, param = line.strip().split(' ')

            log_path_param.append((log_path, param))

    params = []
    accuracies = []

    #sort the log_path_param list by param
    log_path_param.sort(key=lambda x: x[1])
    
    #plot the accuracies of the models w.r.t. to the hyperparameter
    for (log_path, param) in log_path_param:

        # Open the setup.log file in read mode
        with open(log_path, 'r') as file:
            # Read the contents of the file
            contents = file.read()

            # Use regex to find the number after 'validation accuracy' in each line
            pattern = r'validation accuracy\s+(\d+\.\d+)' 
            matches = re.findall(pattern, contents)

            #take the max accuracy
            max_accuracy = max([float(match) for match in matches])

            #add the accuracy to the list
            params.append(float(param))
            accuracies.append(max_accuracy/100)
        
    #plot the accuracies without line joining the points
    plt.plot(log10(params), accuracies, 'o')

    #plot the accuracy of the dense model with a dashed line
    plt.axhline(y=0.9924, color='b', linestyle='--', label='Dense model')

    #plot the accuracy of the BiP model with a dashed line
    plt.axhline(y=0.9926, color='r', linestyle='--', label='BiP with k = 0.1')

    # Add labels to the axes
    plt.xlabel('log10 of the hyperparameter')
    plt.ylabel('Validation accuracy')

    # Add a title
    plt.title('Validation accuracy of the models')
    plt.legend()

    # Save the plot
    save_directory = 'plots'
    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

    save_path = os.path.join(save_directory, 'accuracies.png')
    plt.savefig(save_path)

    print('Plot saved.')


if __name__ == '__main__':
    main()