import torch
print(torch.version.cuda)       # CUDA version PyTorch was built with
print(torch.cuda.is_available())  # Check if GPU is accessible
import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))

