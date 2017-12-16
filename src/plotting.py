import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from main import Network

if __name__ == '__main__':
    # within the network are not only the weights, but also the convergence history
    network = torch.load('../weights/weights_EN_full_dataset_labeled/latest_weights')

    current_path = os.path.dirname(os.path.realpath(__file__))
    savedir = current_path + '/../results/EN_full_dataset_labeled'

    plt.title('arc loss for the 200th datapoint')
    plt.plot(network.arc_loss_particular, 'b-')
    plt.savefig(savedir + '/arc_loss_particular.png')
    plt.clf()

    plt.title('label loss for the 200th datapoint')
    plt.plot(network.label_loss_particular, 'g-')
    plt.savefig(savedir + '/label_loss_particular.png')
    plt.clf()

    plt.title('total loss for the 200th datapoint')
    plt.plot(network.total_loss_particular, 'r-')
    plt.savefig(savedir + '/total_loss_particular.png')
    plt.clf()

    plt.title('arc loss of the dataset')
    plt.plot(network.arc_loss, 'b-')
    plt.savefig(savedir + '/arc_loss.png')
    plt.clf()

    plt.title('label loss of the dataset')
    plt.plot(network.label_loss, 'g-')
    plt.savefig(savedir + '/label_loss.png')
    plt.clf()

    plt.title('total loss of the dataset')
    plt.plot(network.total_loss, 'r-')
    plt.savefig(savedir + '/total_loss.png')
    plt.clf()
