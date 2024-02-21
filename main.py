import argparse

import numpy
import yaml
from PIL import Image
from matplotlib import pyplot

from nets import nn


def load_image(filename):
    with open(filename, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')
    return numpy.array(image)


def show_mask(mask, ax):
    color = numpy.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def save_mask(args, image, masks):
    pyplot.figure(figsize=(10, 10))
    pyplot.imshow(image)
    for mask in masks:
        show_mask(mask, pyplot.gca())
    pyplot.axis("off")
    pyplot.savefig(args.output_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"Saved in {args.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_mask", action="store_true")
    parser.add_argument("--image_path", default="./demo/cat.jpg", type=str)
    parser.add_argument("--output_path", default="./demo/cat.png", type=str)

    parser.add_argument("--mode", default="point", choices=["point", "box"], type=str)
    parser.add_argument("--point", default=None, type=str)
    parser.add_argument("--box", default=None, type=str)

    args = parser.parse_args()

    # build model
    model = nn.build_sam_l2()
    model.cuda()
    model.eval()

    # load image
    image = load_image(args.image_path)
    shape = image.shape
    print(f"Image Size: W={shape[1]}, H={shape[0]}")

    if args.mode == "point":
        args.point = yaml.safe_load(args.point or f"[[{shape[1] // 2},{shape[0] // 2},{1}]]")
        point_coords = [(x, y) for x, y, _ in args.point]
        point_labels = [l for _, _, l in args.point]

        model.set_image(image)
        masks, _, _ = model.predict(point_coords=numpy.array(point_coords),
                                    point_labels=numpy.array(point_labels),
                                    multi_mask_output=args.multi_mask, )
        save_mask(args, image, masks)
    elif args.mode == "box":
        args.box = yaml.safe_load(args.box)
        model.set_image(image)
        masks, _, _ = model.predict(point_coords=None,
                                    point_labels=None,
                                    box=numpy.array(args.box),
                                    multi_mask_output=args.multi_mask, )
        save_mask(args, image, masks)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
