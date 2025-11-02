import tensorflow as tf
import platform

print(f"platform:\n{platform.machine()}")

print(f"tensor flow version: {tf.__version__}")

physical_devices = tf.config.list_physical_devices()
print(f"physical devices:\n{physical_devices}")

gpu_devices = tf.config.list_physical_devices('GPU')
print(f"gpu devices:\n{gpu_devices}")
