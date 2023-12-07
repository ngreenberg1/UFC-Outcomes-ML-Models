import tensorflow as tf

# Enable Metal Pluggable Device
#tf.config.experimental.enable_metal()

# Check the number of available GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print(len(physical_devices))