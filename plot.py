{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from train import *\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def plot_accuracy():\n",
    "    acc = history.history['acc']\n",
    "    val_acc = history.history['val_acc']\n",
    "    \n",
    "    epochs = range(len(acc))\n",
    "    plt.plot(epochs, acc, 'b', label='Training acc')\n",
    "    plt.plot(epochs, val_acc, 'r', label='Validation acc')\n",
    "    plt.title('Training and validation accuracy')\n",
    "    plt.legend()\n",
    "    plt.figure()\n",
    "\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "\n",
    "def plot_losses():\n",
    "    loss = history.history['loss']\n",
    "    val_loss = history.history['val_loss']\n",
    "    epochs = range(len(loss))\n",
    "    plt.plot(epochs, loss, 'b', label='Training loss')\n",
    "    plt.plot(epochs, val_loss, 'r', label='Validation loss')\n",
    "    plt.title('Training and validation loss')\n",
    "    plt.legend()\n",
    "    plt.figure()   \n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
