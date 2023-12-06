import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint
import time


dataframe = pd.read_csv("preprocessed_data.csv", header=0)

dataframe['Winner'] = dataframe['Winner'].apply(lambda x: 1 if x == 'Red' else 0).astype(int)
dataframe['title_bout'] = dataframe['title_bout'].apply(lambda x: 1 if x == True else 0).astype(int)


# Separate features and target variable
X = dataframe.drop('Winner', axis=1)
y = dataframe['Winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=.2, random_state=42)


'''
This model consists of a 1D convolutional layer followed by max pooling, 
a flattening layer, and two fully connected layers for binary classification. The 
ReLU activation function is used in the convolutional and dense layers, while 
the output layer uses a sigmoid activation to produce the final probability prediction.

'''
model = models.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(layers.MaxPooling1D(2))
# Add more layers as needed
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification (winner/loser)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 15

start_time = time.time()

checkpoint = ModelCheckpoint("cnn_model.h5", save_best_only=True, monitor= 'val_loss', mode='min', verbose=1)
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validate, y_validate), callbacks = [checkpoint])

print()
end_time = time.time()
training_time = end_time - start_time
print()

best_model = models.load_model("cnn_model.h5")

test_loss, test_acc = best_model.evaluate(X_test, y_test)

print(f'Test accuracy: {test_acc}')
print(print(f"Training time: {training_time} seconds"))



