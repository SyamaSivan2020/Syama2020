{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from model import *\n",
    "from callbacks import *\n",
    "\n",
    "\n",
    "training_dir = 'imagecl/train'\n",
    "epochs_num = 90\n",
    "\n",
    "\n",
    "training_data, validation_data = import_data(train_dir = training_dir)\n",
    "\n",
    "model = build_model()\n",
    "\n",
    "compile_model(model, optimzation_algorithm = 'Adam', learning_rate = 0.0001)\n",
    "\n",
    "\n",
    "history = model.fit_generator(training_data,\n",
    "            validation_data = validation_data,\n",
    "            steps_per_epoch = 25,\n",
    "            epochs = epochs_num,\n",
    "            validation_steps = 6, \n",
    "            shuffle = True, \n",
    "            verbose = 2, \n",
    "            callbacks=callbacks)"
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
