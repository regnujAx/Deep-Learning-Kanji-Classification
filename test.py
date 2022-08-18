import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, test_images, test_labels, batch_size):
    print("\nEvaluate model on test data:")
    results = model.evaluate(test_images, test_labels, batch_size=batch_size)
    print("Test loss: {:.4f}".format(results[0]))
    print("Test accuracy: {:.4f}".format(results[1]))
    print("\n\n")
    
    # Get the current date and time
    today = datetime.now()
    currentDateTime = today.strftime("%b-%d-%Y-%H-%M-%S")

    f = open("evaluation.txt", "a")
    f.write(currentDateTime + ":\n")
    f.write("The test loss is: " + str(results[0]) + "\n")
    f.write("The test accuracy is: " + str(results[1]) + "\n\n")

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
    f.write("The Precision is: " + str(precision) + "\n")
    f.write("The Recall is: " + str(recall) + "\n")
    f.write("The F1-score is: " + str(f1) + "\n\n")
    f.close()


def plot_accuracy_and_loss(history, number):
    # Plot accuracy and loss
    plt.plot(history.history['loss'], label = 'Training loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.suptitle('Loss', fontsize=20)
    plt.savefig(f'loss_{number}.png')
    plt.clf()

    plt.plot(history.history['accuracy'], label = 'Training accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.suptitle('Accuracy', fontsize=20)
    plt.savefig(f'accuracy_{number}.png')
    plt.clf()
