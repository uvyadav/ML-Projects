import numpy as np 
import pandas as pd 
import json

# importing datasets
train = pd.read_json("train.json")
test = pd.read_json("test.json")

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train["is_iceberg"])

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)

from keras.models import Sequential
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout
# building a simple CNN using keras

model = Sequential()
model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, validation_split=.25, batch_size = 1)

prediction = model.predict(X_test, verbose=1)
score = model.evaluate(X_train, y_train, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])

submit_df = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.flatten()})
submit_df.to_csv("./submission.csv", index=False)
