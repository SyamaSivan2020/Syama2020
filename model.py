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
    "image_height = 300\n",
    "image_width = 300\n",
    "\n",
    "batch_size = 32\n",
    "\n",
    "def import_data(train_dir, batch_size = batch_size):\n",
    "\n",
    "    \n",
    "    \n",
    "    train_generator = ImageDataGenerator(rescale = 1. / 255, \n",
    "                                        rotation_range=20,\n",
    "                                        width_shift_range=0.2,\n",
    "                                        height_shift_range=0.2,\n",
    "                                        horizontal_flip=True,\n",
    "                                        fill_mode = 'nearest',\n",
    "                                        validation_split=0.2)\n",
    "\n",
    "    train_data = train_generator.flow_from_directory(\n",
    "        train_dir, \n",
    "        subset='training',\n",
    "        target_size = (image_height, image_width), \n",
    "        batch_size = batch_size, \n",
    "        class_mode = 'categorical', \n",
    "        shuffle = False\n",
    "\n",
    "    )\n",
    "\n",
    "    val_data = train_generator.flow_from_directory(\n",
    "        train_dir,\n",
    "        subset='validation',\n",
    "        target_size = (image_height, image_width), \n",
    "        batch_size = batch_size, \n",
    "        class_mode = 'categorical', \n",
    "        shuffle = False\n",
    "    )\n",
    "\n",
    "    return train_data, val_data\n",
    "\n",
    "\n",
    "\n",
    "def build_model(classes_num = 3):\n",
    "\n",
    "    pretrained_model = tf.keras.applications.VGG16(input_shape = (image_height, image_height, 3), include_top = False, weights = 'imagenet')\n",
    "\n",
    "\n",
    "\n",
    "    for layer in pretrained_model.layers[:-4]:\n",
    "        layer.trainable = False\n",
    "\n",
    "\n",
    "    for layer in pretrained_model.layers:\n",
    "        print(layer, layer.trainable)\n",
    "\n",
    "    model = tf.keras.models.Sequential()\n",
    "\n",
    "    model.add(pretrained_model)\n",
    "    model.add(tf.keras.layers.Flatten())\n",
    "    model.add(tf.keras.layers.Dense(512, activation = 'relu'))\n",
    "    model.add(tf.keras.layers.Dense(512, activation = 'relu'))\n",
    "    model.add(tf.keras.layers.Dropout(0.2))\n",
    "    model.add(tf.keras.layers.Dense(classes_num, activation = 'softmax'))\n",
    "\n",
    "    return model\n",
    "\n",
    "\n",
    "def compile_model(model, optimzation_algorithm = 'Adam', learning_rate = 0.0001):\n",
    "    \n",
    "    if optimzation_algorithm == 'Adam':\n",
    "        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)\n",
    "    elif optimzation_algorithm == 'RMSprop':\n",
    "        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)\n",
    "    \n",
    "    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy',  metrics = ['acc'])"
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