import csv
import cv2
import numpy as np
import sklearn

# Read the data images
samples = []
corrections = [0.0, 0.25, -0.25] # used for center, left, right image
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header
    next(reader)
    for line in reader:
        for i in range(3): # Add the 3 images with their corrections, so they can be suffled later
            angle = float(line[3]) + corrections[i]
            samples.append([line[i], angle])

# Split the samples for training and validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                    source_path = batch_sample[0]
                    filename = source_path.split('/')[-1]
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = batch_sample[1]
                    measurements.append(measurement)
                    # Flip Image and add it to the batch also
                    images.append(cv2.flip(image, 1))
                    measurements.append(measurement * -1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

# Preprocessing 
model = Sequential()
# Crop first to work with less data
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# NVIDIA model
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model.compile(loss='mse', optimizer='adam')
# Samples per epoch is * 2, cause for each batch we return double images (normal and flipped)
history_object = model.fit_generator(train_generator, \
            samples_per_epoch=(len(train_samples) * 2), \
            validation_data=validation_generator, \
            nb_val_samples=(len(validation_samples) * 2), \
            nb_epoch=3, verbose=1)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
