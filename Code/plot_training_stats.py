# for visualisation of graphs or confision matrix
import matplotlib.pyplot as plt

def plot_model_accuracy(history, figsize=(16,10)):
    # Visualisation validation accuracy
    plt.figure(figsize=figsize)
    plt.plot(history.history['acc'], "b--")
    plt.plot(history.history['val_acc'],"r--")
    plt.title('Genauigkeit Training/Test', fontsize=20)
    plt.ylabel('Genauigkeit', fontsize=19)
    plt.xlabel('Epoche', fontsize=19)
    plt.legend(['Training','Test'],loc='lower right', fontsize=14)
    return plt.show()

def plot_model_loss(history, figsize=(16,10)):
    # Visualisation validation loss
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], "b--")
    plt.plot(history.history['val_loss'], "r--")
    plt.title('Fehlerrate Training/Test', fontsize=20)
    plt.ylabel('Fehlerrate', fontsize=17)
    plt.xlabel('Epoche', fontsize=17)
    plt.legend(['Training', 'Test'] , loc='upper right', fontsize=14)
    return plt.show()