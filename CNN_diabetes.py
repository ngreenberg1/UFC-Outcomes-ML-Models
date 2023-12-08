import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
import time

dataframe = pd.read_csv("preprocessed_data.csv", header=0)

dataframe['Winner'] = dataframe['Winner'].apply(lambda x: 1 if x == 'Red' else 0).astype(int)
dataframe['title_bout'] = dataframe['title_bout'].apply(lambda x: 1 if x == True else 0).astype(int)


# Separate features and target variable
X = dataframe.drop('Winner', axis=1)
y = dataframe['Winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=.2, random_state=42)

# Check the number of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)


# Define the strategy
strategy = tf.distribute.MirroredStrategy()

# Create and compile the model inside the strategy scope
with strategy.scope():
    model = models.Sequential()
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(layers.MaxPooling1D(2))
    # Add more layers as needed
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification (winner/loser)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Measure the time taken for training
start_time = time.time()

epochs = 15
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validate, y_validate))

end_time = time.time()
training_time = end_time - start_time

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'Test accuracy: {test_acc}')
print(f"Number of GPUs available: {num_gpus}")
print(f"Training time: {training_time} seconds")
