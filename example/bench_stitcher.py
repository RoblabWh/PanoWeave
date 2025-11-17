#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from time import time
from argparse import ArgumentParser, ArgumentError
from panoweave import Stitcher


def readInputs(inputs):
    imgs = list()
    ret = True
    for input in inputs:
        cap = cv.VideoCapture(input)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {input}")
        ret, img = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame from: {input}")
        imgs.append(img)
    return ret, imgs


def main():
    parser = ArgumentParser(prog="Video Stitcher Example")
    parser.add_argument("inputs", nargs="+", type=str, help="Path to input videos")
    parser.add_argument(
        "--calibration",
        "--calib",
        "-c",
        type=str,
        help="Path to basalt calibration.json for camera calibration",
        required=True,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3200,
        help="Output resolution width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1600,
        help="Output resolution height in pixels",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=1000,
        help="Number of iterations to run",
    )
    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="Skip the first iteration (removes lazy loading time)",
    )
    parser.add_argument(
        "--simulate-depth",
        action="store_true",
        help="Simulate depth data",
    )
    try:
        args = parser.parse_args()
    except ArgumentError as e:
        print(e.message)
        parser.print_help()
        return 1

    print(f"OpenCL Support: {cv.ocl.haveOpenCL()}")
    if cv.ocl.haveOpenCL():
        device = cv.ocl.Device.getDefault()
        print(
            f"Device: {device.name()}\n  Available: {device.available()}\n  Compiler:  {device.compilerAvailable()}\n  Linker:    {device.linkerAvailable()}\n"
        )

    stitcher = Stitcher(args.calibration)
    stitcher.resolution(args.width, args.height)
    stitcher.setDepth(1)

    ret, imgs = readInputs(args.inputs)
    if not ret:
        print("Failed to read input images.")
        return 1

    for i, img in enumerate(imgs):
        cv.namedWindow(f"Input{i}", cv.WINDOW_NORMAL)
        cv.imshow(f"Input{i}", img)
    cv.namedWindow("Output", cv.WINDOW_NORMAL)

    if args.skip_first:
        stitcher.stitch(imgs)

    print(f"Resolution: {args.width}x{args.height}\nIterations: {args.iterations}")
    durations = list()
    duration = 0
    for i in range(args.iterations):
        start = time()
        pano = stitcher.stitch(imgs)
        end = time()
        d = end - start
        duration += d
        durations.append(d)

        if args.simulate_depth:
            stitcher.setDepth((i % 100 + 1) / 10)

        cv.imshow("Output", pano)
        if cv.waitKey(1) == 27:
            break

    mean = np.mean(durations)
    stddev = np.std(durations)

    print(
        f"Benchmark:\n  Samples: {len(durations)}\n  Mean:    {mean:.6f}s\t{1 / mean:.6f}Hz\n  StdDev:  {stddev:.6f}s"
    )


if __name__ == "__main__":
    main()
