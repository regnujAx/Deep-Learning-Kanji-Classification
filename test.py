import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, test_images, test_labels, batch_size):
    print("Evaluate model on test data...")

    results = model.evaluate(test_images, test_labels, batch_size=batch_size)
    print("The test loss is:", results[0])
    print("The test accuracy is:", results[1])
    print("\n\n")

    # Get the current date and time
    today = datetime.now()
    currentDateTime = today.strftime("%b-%d-%Y-%H-%M-%S")

    file = open("evaluation.txt", "a")
    file.write(f"{currentDateTime}:\n")
    file.write(f"The test loss is: {str(results[0])}\n")
    file.write(f"The test accuracy is: {str(results[1])}\n\n")

    predictions = model.predict(test_images, batch_size=batch_size)

    # Decode predicted and truth labels as one vector
    predicted_labels = np.argmax(predictions, axis=1)
    truth_labels = np.argmax(test_labels, axis=1)

    # Calculate and print f1, precision and recall scores
    precision = precision_score(truth_labels, predicted_labels, average="macro", zero_division=0)
    recall = recall_score(truth_labels, predicted_labels, average="macro", zero_division=0)
    f1 = f1_score(truth_labels, predicted_labels, average="macro", zero_division=0)
    print("The Precision is:", precision)
    print("The Recall is:", recall)
    print("The F1-score is:", f1)
    file.write(f"The Precision is: {str(precision)}\n")
    file.write(f"The Recall is: {str(recall)}\n")
    file.write(f"The F1-score is: {str(f1)}\n\n")
    file.close()


def plot_accuracy_and_loss(history, label):
    print("Plot loss and accuracy...")

    plt.plot(history.history["loss"], label = "Training loss")
    plt.plot(history.history["val_loss"], label = "Validation loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.suptitle("Loss", fontsize=20)
    plt.savefig(f"loss_{label}.png")
    plt.clf()

    plt.plot(history.history["accuracy"], label = "Training accuracy")
    plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.suptitle("Accuracy", fontsize=20)
    plt.savefig(f"accuracy_{label}.png")
    plt.clf()


def plot_predictions(model, test_images, test_labels, shape, label, channels=1):
    print("Plot some output predictions...")

    n = 8
    save_name = "predictions.png"
    if label != "":
        save_name = f"predictions_{label}.png"

    # Generate n² random output predictions
    random_idx = np.random.randint(0, test_images.shape[0], n**2)

    random_set = test_images[random_idx]
    random_labels = test_labels[random_idx].argmax(axis=1).reshape(n, n)
    pred_labels = model.predict(random_set).argmax(axis=1).reshape(n, n)

    random_set = random_set.reshape(n, n, shape[0], shape[1], channels)

    # Create an n x n grid
    fig, ax = plt.subplots(n, n, figsize=(15, 15))
    fig.suptitle("Test set predictions (truth label/prediction label)", fontsize=25)

    for i in range(random_set.shape[0]):
        for j in range(random_set.shape[1]):
            example = random_set[i, j]
            ax[i, j].imshow(example, cmap="gray")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            color = ("green" if random_labels[i, j] == pred_labels[i, j] else "red")
            random_label = random_labels[i, j]
            pred_label = pred_labels[i, j]
            ax[i, j].set_title(f"{random_label}/{pred_label}", color=color)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(save_name)
    # plt.show()
