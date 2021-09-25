# python3.7
"""Contains the base class for generator."""

import os
import numpy as np
import torch


class BaseGenerator(object):
  """Base class for generator used in GAN variants.

  NOTE: The model should be defined with pytorch, and only used for inference.
  """

  def __init__(self, model_name, logger=None):
    self.model_name = model_name

  def check_attr(self, attr_name):
    """Checks the existence of a particular attribute.

    Args:
      attr_name: Name of the attribute to check.

    Raises:
      AttributeError: If the target attribute is missing.
    """
    if not hasattr(self, attr_name):
      raise AttributeError(
          f'`{attr_name}` is missing for model `{self.model_name}`!')

  def build(self):
    """Builds the graph."""
    raise NotImplementedError(f'Should be implemented in derived class!')

  def load(self):
    """Loads pre-trained weights."""
    raise NotImplementedError(f'Should be implemented in derived class!')

  def convert_tf_model(self, test_num=10):
    """Converts models weights from tensorflow version.

    Args:
      test_num: Number of images to generate for testing whether the conversion
        is done correctly. `0` means skipping the test. (default 10)
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def sample(self, num):
    """Samples latent codes randomly.

    Args:
      num: Number of latent codes to sample. Should be positive.

    Returns:
      A `numpy.ndarray` as sampled latend codes.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def preprocess(self, latent_codes):
    """Preprocesses the input latent code if needed.

    Args:
      latent_codes: The input latent codes for preprocessing.

    Returns:
      The preprocessed latent codes which can be used as final input for the
        generator.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_sample(self, num):
    """Wraps functions `sample()` and `preprocess()` together."""
    return self.preprocess(self.sample(num))

  def synthesize(self, latent_codes):
    """Synthesizes images with given latent codes.

    NOTE: The latent codes should have already been preprocessed.

    Args:
      latent_codes: Input latent codes for image synthesis.

    Returns:
      A dictionary whose values are raw outputs from the generator.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def get_value(self, tensor):
    """Gets value of a `torch.Tensor`.

    Args:
      tensor: The input tensor to get value from.

    Returns:
      A `numpy.ndarray`.

    Raises:
      ValueError: If the tensor is with neither `torch.Tensor` type or
        `numpy.ndarray` type.
    """
    if isinstance(tensor, np.ndarray):
      return tensor
    if isinstance(tensor, torch.Tensor):
      return tensor.to(self.cpu_device).detach().numpy()
    raise ValueError(f'Unsupported input type `{type(tensor)}`!')

  def postprocess(self, images):
    """Postprocesses the output images if needed.

    This function assumes the input numpy array is with shape [batch_size,
    channel, height, width]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The return images are with shape
    [batch_size, height, width, channel]. NOTE: The channel order of output
    image will always be `RGB`.

    Args:
      images: The raw output from the generator.

    Returns:
      The postprocessed images with dtype `numpy.uint8` with range [0, 255].

    Raises:
      ValueError: If the input `images` are not with type `numpy.ndarray` or not
        with shape [batch_size, channel, height, width].
    """
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')

    images_shape = images.shape
    if len(images_shape) != 4 or images_shape[1] not in [1, 3]:
      raise ValueError(f'Input should be with shape [batch_size, channel, '
                       f'height, width], where channel equals to 1 or 3. '
                       f'But {images_shape} is received!')
    images = (images - self.min_val) * 255 / (self.max_val - self.min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    if self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]

    return images

  def easy_synthesize(self, latent_codes, **kwargs):
    """Wraps functions `synthesize()` and `postprocess()` together."""
    outputs = self.synthesize(latent_codes, **kwargs)
    if 'image' in outputs:
      outputs['image'] = self.postprocess(outputs['image'])

    return outputs

  def get_batch_inputs(self, latent_codes):
    """Gets batch inputs from a collection of latent codes.

    This function will yield at most `self.batch_size` latent_codes at a time.

    Args:
      latent_codes: The input latent codes for generation. First dimension
        should be the total number.
    """
    total_num = latent_codes.shape[0]
    for i in range(0, total_num, self.batch_size):
      yield latent_codes[i:i + self.batch_size]