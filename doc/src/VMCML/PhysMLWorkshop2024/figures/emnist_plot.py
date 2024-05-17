import emnist
import matplotlib.pyplot as plt
"""
plot the first example from each class in the EMNIST balanced dataset
"""

(train_images, train_labels) = emnist.extract_training_samples('balanced')

# number of classes
num_classes = 47
nrows = 3

fig, axs = plt.subplots(3, num_classes//nrows + 1, figsize=(16, 3))

for i in range(num_classes):
    ax = axs.flat[i]
    ax.imshow(train_images[train_labels == i][0], cmap='gray')
    ax.axis('off')

for ax in axs.flat:
    ax.axis('off')

plt.subplots_adjust(wspace=0.0, hspace=0.0)
#plt.tight_layout()
fig.savefig('emnist.pdf',bbox_inches='tight')


