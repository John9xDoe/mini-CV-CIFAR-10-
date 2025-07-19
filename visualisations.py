import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def show_image(img_idx, data_images, data_names):
        img_flat = data_images[img_idx]

        r = img_flat[0:1024].reshape(32, 32)
        g = img_flat[1024:2048].reshape(32, 32)
        b = img_flat[2048:].reshape(32, 32)

        img = np.stack([r, g, b], axis=2)
        plt.imshow(img.astype('uint8'))
        # plt.axis('off')
        plt.title(f'ID: {img_idx}\nClass: {data_names[img_idx]}')
        plt.show()

    @staticmethod
    def plot_epochs_currency(epochs_data, accuracy_data):
        plt.plot(epochs_data, accuracy_data, label='Зависимость точности от количества эпох')
        plt.xlabel("Эпоха")
        plt.ylabel("Точность")
        plt.grid(True)
        plt.legend()
        plt.savefig(f'plots/plot_{len(epochs_data)}_epochs.png')
        plt.show()