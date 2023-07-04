import argparse
import matplotlib.pyplot as plt
from torch import load
from numpy import linspace, zeros, log
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

    checkpoint_label = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Split the line into two words
            log_path, param = line.strip().split(' ')

            log_path_param.append((log_path, param))

    
    common_length = None

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
            params.append(log(float(param)))
            accuracies.append(max_accuracy)
        
    #plot the accuracies
    plt.plot(log(params), accuracies)

    #plot the accuracy of the dense model with a dashed line
    plt.axhline(y=0.992, color='b', linestyle='--', label='Dense model')

    # Add a title
    plt.title('Validation accuracy of the models')

    # Save the plot
    save_directory = 'plots'
    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

    save_path = os.path.join(save_directory, 'accuracies.png')
    plt.savefig(save_path)

    print('Plot saved.')


if __name__ == '__main__':
    main()