{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import os \n",
    "from matplotlib import pyplot as plt\n",
    "from train import *\n",
    "from random import randrange\n",
    "\n",
    "\n",
    "test_imgs_dir= './test/'\n",
    "\n",
    "\n",
    "def get_classes():\n",
    "    classes = training_data.class_indices\n",
    "    classes = {v: k for k, v in classes.items()}\n",
    "\n",
    "    return classes\n",
    "\n",
    "\n",
    "def classify_batch(batch = 20):\n",
    "\n",
    "    classes = get_classes()\n",
    "\n",
    "    img_list = os.listdir(test_imgs_dir)\n",
    "    images_number = len(img_list)\n",
    "\n",
    "    index = randrange(0,images_number,1)\n",
    "\n",
    "    batch_holder = np.zeros((batch, image_height, image_width, 3))\n",
    "    for i,img in enumerate(img_list[index: index + batch]):\n",
    "        img = tf.keras.preprocessing.image.load_img(os.path.join(test_imgs_dir,img), target_size=(image_height,image_height))\n",
    "        batch_holder[i, :] = img\n",
    "\n",
    "\n",
    "    result = model.predict_classes(batch_holder)\n",
    "    \n",
    "    fig = plt.figure(figsize=(25, 25))\n",
    "    \n",
    "    for i,img in enumerate(batch_holder):\n",
    "        fig.add_subplot(4,5, i+1)\n",
    "        plt.title(classes[result[i]])\n",
    "        plt.axis('off')\n",
    "        plt.imshow(img/256.) \n",
    "\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "\n",
    "def classify_image(image):\n",
    "    classes = get_classes()\n",
    "\n",
    "    img = tf.keras.preprocessing.image.load_img(image, target_size=(image_height,image_width))\n",
    "    img = np.array(img)\n",
    "    img = img.reshape(1, image_height,image_width, 3)\n",
    "    pred = np.argmax(model.predict(img))\n",
    "\n",
    "    print(classes[pred])"
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
