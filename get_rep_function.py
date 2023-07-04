import argparse
import matplotlib.pyplot as plt
from torch import load
from numpy import linspace, zeros
import os

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Superimpose the repartition functions of the popup scores')

    # Add command line arguments
    parser.add_argument('--file', type=str, help='Text file with all the paths to the checkpoints')
    parser.add_argument('--n', type=int, help='Number of points to plot', default=1000)
    #parser.add_argument('--save-path', type=str, help='Path to the image of the plot')


    # Parse the command line arguments
    args = parser.parse_args()

    # Access the argument values
    file_path = args.file
    n_points = args.n

    checkpoint_label = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Split the line into two words
            checkpoint_path, label = line.strip().split(' ')

            checkpoint_label.append((checkpoint_path, label))

    
    common_length = None
    
    #plot the repartition functions of each mask on the same plot
    for (checkpoint_path, label) in checkpoint_label:
        checkpoint = load(checkpoint_path)
        model_dict = checkpoint['state_dict']

        mask_list = []
        for (name, tensor) in model_dict.items():
            if 'popup_scores' in name:
                #retrieve the params of the layer
                mask_list.extend(tensor.view(-1).detach().tolist())
        
        mask_length = len(mask_list)
        if common_length is not None and mask_length != common_length:
            raise ValueError(f'Length of mask for {checkpoint} is {mask_length} while it should be {common_length}.')
        mask_list.sort()
        
        x = linspace(0, mask_list[-1], n_points)
        probs = zeros(n_points)
        pointer = 0
        for i in range(n_points):
            while pointer < mask_length and mask_list[pointer] < x[i]:
                pointer += 1
            probs[i] = pointer / mask_length
        
        plt.plot(x, probs, label=label)

    plt.legend()
    # Save the plot
    save_directory = 'plots'
    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

    save_path = os.path.join(save_directory, 'rep_function.png')
    plt.savefig(save_path)

    print('Plot saved.')


if __name__ == '__main__':
    main()