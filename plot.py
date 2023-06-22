import matplotlib.pyplot as plt
from torch import load

#given a checkpoint, we want to plot the repartition function of the popup scores 
#and save the image of the plot in images/plot.png

path = 'trained_models/MNIST_mnist_model_PenBip1_L100/prune/latest_exp/checkpoint/model_best.pth.tar'

checkpoint = load(path)
model_dict = checkpoint['state_dict']

print('Model loaded.')


#retrieve the popup scores as a list
mask_list = []
for (name, module) in model_dict.items():
    if 'popup_scores' in name:
        print(type(module))
        #retrieve the params of the layer
        attr = getattr(module, 'popup_scores')
        #add the params to the list
        mask_list.extend(attr.view(-1).detach().tolist())


#print the lenght of the list 

print(f'Number of popup scores: {len(mask_list)}')


#plot the repartition function of the popup scores
mask_list.sort()
n_points = 1000
mask_length = len(mask_list)

x = np.linspace(0, mask_list[-1], n_points)
probs = np.zeros(n_points)
pointer = 0
for i in range(n_points):
    while pointer < mask_length and mask_list[pointer] < x[i]:
        pointer += 1
    probs[i] = pointer / mask_length

plt.plot(x, probs)

#save the plot
plt.savefig('images/plot.png')

print('Plot saved.')



