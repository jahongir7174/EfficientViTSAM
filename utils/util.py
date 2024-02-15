import copy

import numpy
import torch
from torch.nn import functional
from torchvision.transforms.functional import resize, to_pil_image


def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


class SamResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image):
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return numpy.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(oldh, oldw, long_side_length):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return list([newh, neww])


class SamPad:
    def __init__(self, size, fill=0, pad_mode="corner"):
        self.size = size
        self.fill = fill
        self.pad_mode = pad_mode

    def __call__(self, image):
        h, w = image.shape[-2:]
        th, tw = self.size, self.size
        assert th >= h and tw >= w
        if self.pad_mode == "corner":
            image = functional.pad(image, (0, tw - w, 0, th - h), value=self.fill)
        else:
            raise NotImplementedError
        return image


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length) -> None:
        self.target_length = target_length

    def apply_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return numpy.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords, original_size):
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size):
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image):
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return functional.interpolate(image, target_size, mode="bilinear", align_corners=False, antialias=True)

    def apply_coords_torch(self, coords, original_size):
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = copy.deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(self, boxes, original_size):
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh, oldw, long_side_length):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return list([newh, neww])
