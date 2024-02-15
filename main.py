import argparse

import numpy
import yaml
from PIL import Image

from nets import nn


def concatenate(images, axis=1, pad=20):
    shape_list = [image.shape for image in images]
    max_h = max([shape[0] for shape in shape_list]) + pad * 2
    max_w = max([shape[1] for shape in shape_list]) + pad * 2

    for i, image in enumerate(images):
        canvas = numpy.zeros((max_h, max_w, 3), dtype=numpy.uint8)
        h, w, _ = image.shape
        crop_y = (max_h - h) // 2
        crop_x = (max_w - w) // 2
        canvas[crop_y: crop_y + h, crop_x: crop_x + w] = image
        images[i] = canvas

    image = numpy.concatenate(images, axis=axis)
    return image


def draw_binary_mask(raw_image, binary_mask, mask_color=(0, 0, 255)):
    color_mask = numpy.zeros_like(raw_image, dtype=numpy.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = numpy.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = numpy.asarray(canvas, dtype=numpy.uint8)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_mask", action="store_true")
    parser.add_argument("--image_path", default="zidane.jpg", type=str)
    parser.add_argument("--output_path", default="zidane.png", type=str)

    parser.add_argument("--mode", default="point", choices=["point", "box"], type=str)
    parser.add_argument("--point", default=None, type=str)
    parser.add_argument("--box", default=None, type=str)

    args = parser.parse_args()

    # build model
    model = nn.build_sam_l2()
    model.cuda()
    model.eval()

    # load image
    raw_image = numpy.array(Image.open(args.image_path).convert("RGB"))
    h, w, _ = raw_image.shape
    print(f"Image Size: W={w}, H={h}")

    if args.mode == "point":
        args.point = yaml.safe_load(args.point or f"[[{w // 2},{h // 2},{1}]]")
        point_coords = [(x, y) for x, y, _ in args.point]
        point_labels = [l for _, _, l in args.point]

        model.set_image(raw_image)
        masks, _, _ = model.predict(point_coords=numpy.array(point_coords),
                                    point_labels=numpy.array(point_labels),
                                    multi_mask_output=args.multi_mask, )
        plots = [draw_binary_mask(raw_image, binary_mask, (0, 0, 255)) for binary_mask in masks]
        plots = concatenate(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)
    elif args.mode == "box":
        args.box = yaml.safe_load(args.box)
        model.set_image(raw_image)
        masks, _, _ = model.predict(point_coords=None,
                                    point_labels=None,
                                    box=numpy.array(args.box),
                                    multi_mask_output=args.multi_mask, )
        plots = [draw_binary_mask(raw_image, binary_mask, (0, 0, 255)) for binary_mask in masks]
        plots = concatenate(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
