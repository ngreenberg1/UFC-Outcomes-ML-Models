import tensorflow as tf
import ssl

# Disable SSL verification (if needed)
ssl._create_default_https_context = ssl._create_unverified_context

# Load CIFAR-100 dataset
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# Define a MirroredStrategy for distributing training across multiple CPUs
strategy = tf.distribute.MirroredStrategy()

# Create the model within the strategy's scope
with strategy.scope():
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=32)
