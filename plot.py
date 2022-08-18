import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(model, test_images, test_labels):
    n = 8
    
    # Generate n² random output predictions
    random_idx = np.random.randint(0, test_images.shape[0], n**2)

    random_set = test_images[random_idx]
    random_labels = test_labels[random_idx].argmax(axis=1).reshape(n, n)
    pred_labels = model.predict(random_set).argmax(axis=1).reshape(n, n)

    random_set = random_set.reshape(n, n, 64, 64)

    # Create n x n grid
    fig, ax = plt.subplots(n, n, figsize=(15, 15))
    fig.suptitle('Test set predictions (truth/prediction)', fontsize=20)

    for i in range(random_set.shape[0]):
        for j in range(random_set.shape[1]):
            example = random_set[i, j]
            ax[i, j].imshow(example, cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            color = ('green' if random_labels[i, j] == pred_labels[i, j] else 'red')
            ax[i, j].set_title(f"{random_labels[i, j]}/{pred_labels[i, j]}", color=color)

    plt.subplots_adjust(hspace=0.5)
    # plt.show()
    plt.savefig('predictions.png')