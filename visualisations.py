import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, data_images, data_names):
        self.data_images = data_images
        self.data_names = data_names

    def show_image(self, img_idx):
        img_flat = self.data_images[img_idx]

        r = img_flat[0:1024].reshape(32, 32)
        g = img_flat[1024:2048].reshape(32, 32)
        b = img_flat[2048:].reshape(32, 32)

        img = np.stack([r, g, b], axis=2)
        plt.imshow(img.astype('uint8'))
        # plt.axis('off')
        plt.title(f'ID: {img_idx}\nClass: {self.data_names[img_idx]}')
        plt.show()
