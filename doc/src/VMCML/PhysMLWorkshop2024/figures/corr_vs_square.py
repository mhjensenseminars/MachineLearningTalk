import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import matplotlib as mpl
from matplotlib import font_manager

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}\usepackage{amsmath}'
mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(
    fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
)
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False

plt.rcParams["axes.labelsize"] = 36
plt.rcParams["xtick.labelsize"] = 28
plt.rcParams["ytick.labelsize"] = 28

np.random.seed(0)

# load one image from the MNIST dataset
(train_images, _), (_, _) = mnist.load_data()
image = train_images[np.random.randint(0,60000)].astype(np.float32)/255.0

window_size = 14
corner_pos = (3, 5)

# calculate the correlation and square of the top-left window

window = image[corner_pos[0]:window_size,corner_pos[1]:window_size]

# row-wise correlation
corr_R = np.corrcoef(window)
# column-wise correlation
corr_C = np.corrcoef(window.T)
# replace NaN values with 0
corr_R = np.nan_to_num(corr_R)
corr_C = np.nan_to_num(corr_C)
# write 1.0 to the diagonal
np.fill_diagonal(corr_R, 1.0)
np.fill_diagonal(corr_C, 1.0)

# normalize the correlation matrices to [-1, 1]
max_R = np.max(corr_R)
min_R = np.min(corr_R)
max_C = np.max(corr_C)
min_C = np.min(corr_C)

corr_R = (corr_R - min_R) / (max_R - min_R) * 2 - 1
corr_C = (corr_C - min_C) / (max_C - min_C) * 2 - 1

# calculate the square of the top-left window

sq_R = window @ window.conj().T
sq_C = window.conj().T @ window

# normalize the squares to [-1, 1]
max_R = np.max(sq_R)
min_R = np.min(sq_R)
max_C = np.max(sq_C)
min_C = np.min(sq_C)

sq_R = (sq_R - min_R) / (max_R - min_R) * 2 - 1
sq_C = (sq_C - min_C) / (max_C - min_C) * 2 - 1

# plot the correlation and square matrices
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
# draw rectangle around the window
rect = plt.Rectangle(np.array(corner_pos)-0.5, window_size, window_size, edgecolor='r', facecolor='none', lw=1)
axes[0].add_patch(rect)
axes[0].set_title('Image')
axes[0].axis('off')
axes[1].imshow(corr_R, cmap='bwr', vmin=-1, vmax=1)
axes[1].set_title('$C_R$')
axes[1].axis('off')
axes[2].imshow(sq_R, cmap='bwr', vmin=-1, vmax=1)
axes[2].set_title(r'$X X^\dagger$')
axes[2].axis('off')

plt.subplots_adjust(wspace=0.1)
plt.tight_layout()

#plt.show()
fig.savefig('corr_vs_square.png', dpi=300)


