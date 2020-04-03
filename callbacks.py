{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras import Model\n",
    "\n",
    "\n",
    "accuracy_threshold = 0.99\n",
    "patience = 50\n",
    "\n",
    "\n",
    "class callback(tf.keras.callbacks.Callback): \n",
    "    def on_epoch_end(self, epoch, logs={}): \n",
    "        if(logs.get('acc') > accuracy_threshold):\n",
    "            print(\"Reached %2.2f%% accuracy, Training stopped!!\" %(accuracy_threshold*100))\n",
    "            self.model.stop_training = True\n",
    "            self.model.save('model-optimal.h5')\n",
    "        \n",
    "\n",
    "        \n",
    "checkpoint = tf.keras.callbacks.ModelCheckpoint('weights{epoch:03d}.h5', save_weights_only=True, period=10)     \n",
    "reduce_lr = tf.keras.callbacks.ReduceLROnPlateau('val_acc', factor=0.1, patience=10, verbose=1)\n",
    "callbacks = [callback(), checkpoint]"
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
