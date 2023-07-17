import argparse
import matplotlib.pyplot as plt
from torch import load
from numpy import linspace, zeros, load
import re
import os


n = 22338314 // 2
iterations_to_plot = [9 + 10 * i for i in range(10)]

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Do all the individual plots of all the latest_exp in trained_models')

    # Add command line arguments
    #parser.add_argument('--folder', type=str, help='Name of the folder to look into.')
    #parser.add_argument('--n', type=int, help='Number of points to plot', default=1000)
    #parser.add_argument('--save-path', type=str, help='Path to the image of the plot')

    # Parse the command line arguments
    args = parser.parse_args()

    #we loop trhough all the folders in 'trained_models'
    for folder in os.listdir('trained_models'):

        #we continue if the folder does not contain 'prune' or does not have 'RC' in its name
        if 'prune' not in os.listdir(os.path.join('trained_models', folder)) or 'RC' not in folder:
            print('Skipping ' + folder + '.')
            continue

        #we have to go inside the folder 'prune/latest_exp/' in folder
        path = os.path.join('trained_models', folder, 'prune/latest_exp')

        #open the file 'epochs_data.npy' inside folder
        file_path = os.path.join(path, 'epochs_data.npy')

        print('Plots for ' + folder + ' in progress ...')

        l = load(file_path, allow_pickle=True)

        l0_list = []
        l1_list = []
        mini_list = []
        maxi_list = []

        treshold_list = []
        below_treshold_list = []
        over_treshold_list = []
        
        #it depends what method was used to get the info
        t_length = len(l[0])
        if t_length == 5:
            for (l0, l1, mini, maxi, c) in l:
                l0_list.append(l0 / n)
                l1_list.append(l1 / n)
                mini_list.append(mini)
                maxi_list.append(maxi)
                
                s, t = c
                treshold_list = s
                below_treshold_list.append(t)

            #we want five plots : l0, l1, maxi, below_treshold, accuracy during training

            #l0 plot
            plt.plot(l0_list)
            plt.xlabel('Epochs')
            plt.ylabel('l0 / n')
            plt.title('Evolution of l0 / n during training of ' + folder)

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'l0_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #l1 plot
            plt.plot(l1_list)
            plt.xlabel('Epochs')
            plt.ylabel('l1 / n')
            plt.title('Evolution of l1 / n during training of ' + folder)

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'l1_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #maxi plot
            plt.plot(maxi_list)
            plt.xlabel('Epochs')
            plt.ylabel('Maximum element')
            plt.title('Evolution of the maximum element during training of ' + folder)

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'maxi_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #below_treshold plot
            for iteration in iterations_to_plot:
                plt.plot(['{}'.format(exp) for exp in treshold_list], below_treshold_list[iteration], label='Epoch ' + str(iteration))
            plt.xlabel('Treshold')
            plt.ylabel('Ratio of elements below treshold')
            plt.title('Evolution of the ratio of elements below treshold during training of ' + folder)
            plt.legend()

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'below_treshold_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #accuracy plot
            
            log_path = os.path.join(path, 'setup.log')
            # Open the setup.log file in read mode
            epochs = []
            accuracies = []
            with open(log_path, 'r') as file:

                # Read the contents of the file
                contents = file.read()

                # Use regex to find the number after 'validation accuracy' in each line
                pattern = r'validation accuracy\s+(\d+\.\d+)' 
                matches = re.findall(pattern, contents)

                #for every match, append the next epoch to epochs
                #and the accuracy to accuracies

                for i in range(len(matches)):
                    epochs.append(i)
                    accuracies.append(float(matches[i])/100)
            
            plt.plot(epochs, accuracies)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Evolution of the accuracy during training of : ' + folder)

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'accuracy_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            print('Plots done for ' + folder)
        
        if t_length == 6:
            for (l0, l1, mini, maxi, c1, c2) in l:
                l0_list.append(l0 / n)
                l1_list.append(l1 / n)
                mini_list.append(mini)
                maxi_list.append(maxi)
                
                s1, t1 = c1
                treshold_list = s1
                below_treshold_list.append(t1)

                s2, t2 = c2
                treshold_list = s2
                over_treshold_list.append(t2)

            #we want six plots : l0, l1, maxi, below_treshold, over_treshold, accuracy during training

            #l0 plot
            plt.plot(l0_list)
            plt.xlabel('Epochs')
            plt.ylabel('l0 / n')
            plt.title('Evolution of l0 / n during training of ' + folder)

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'l0_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #l1 plot
            plt.plot(l1_list)
            plt.xlabel('Epochs')
            plt.ylabel('l1 / n')
            plt.title('Evolution of l1 / n during training of ' + folder)

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'l1_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #maxi plot
            plt.plot(maxi_list)
            plt.xlabel('Epochs')
            plt.ylabel('Maximum element')
            plt.title('Evolution of the maximum element during training of ' + folder)

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'maxi_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #below_treshold plot
            for iteration in iterations_to_plot:
                plt.plot(['{}'.format(exp) for exp in treshold_list], below_treshold_list[iteration], label='Epoch ' + str(iteration))
            plt.xlabel('Treshold')
            plt.ylabel('Ratio of elements below treshold')
            plt.title('Evolution of the ratio of elements below treshold during training of ' + folder)
            plt.legend()

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'below_treshold_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #over_treshold plot
            for iteration in iterations_to_plot:
                plt.plot(['{}'.format(exp) for exp in treshold_list], over_treshold_list[iteration], label='Epoch ' + str(iteration))
            plt.xlabel('Treshold')
            plt.ylabel('Ratio of elements over treshold')
            plt.title('Evolution of the ratio of elements over treshold during training of ' + folder)
            plt.legend()

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'over_treshold_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            #accuracy plot
            
            log_path = os.path.join(path, 'setup.log')
            # Open the setup.log file in read mode
            epochs = []
            accuracies = []
            with open(log_path, 'r') as file:

                # Read the contents of the file
                contents = file.read()

                # Use regex to find the number after 'validation accuracy' in each line
                pattern = r'validation accuracy\s+(\d+\.\d+)' 
                matches = re.findall(pattern, contents)

                #for every match, append the next epoch to epochs
                #and the accuracy to accuracies

                for i in range(len(matches)):
                    epochs.append(i)
                    accuracies.append(float(matches[i])/100)
            
            plt.plot(epochs, accuracies)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Evolution of the accuracy during training of : ' + folder)

            # Save the plot
            save_directory = 'plots'
            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
            name = 'accuracy_plot_' + folder + '.png'
            save_path = os.path.join(save_directory, name)
            plt.savefig(save_path)

            #clear the plot
            plt.clf()

            print('Plots done for ' + folder)
            
    print('All plots done with success.')

if __name__ == '__main__':
    main()