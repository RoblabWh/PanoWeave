#!/usr/bin/env python3

import cv2 as cv
from math import pi
from argparse import ArgumentParser, ArgumentError
from scipy.spatial.transform import Rotation as R
from panoweave import Stitcher

def readInputs(caps):
    imgs = list()
    ret = True
    for cap in caps:
        ret, img = cap.read()
        if not ret:
            break
        imgs.append(img)
    return ret, imgs

def modifyRotation(tf, x = None, y = None, z = None):
    rpy = R.from_matrix(tf[:3, :3]).as_euler("XYZ", degrees=True)
    if x is not None:
        rpy[0] = x
    if y is not None:
        rpy[1] = y
    if z is not None:
        rpy[2] = z
    return R.from_euler("XYZ", rpy, degrees=True).as_rotvec()

def main():
    parser = ArgumentParser(prog="Video Stitcher Example")
    parser.add_argument("inputs", nargs="+", type=str, help="path to input videos")
    parser.add_argument("--calibration", "--calib", "-c", type=str, help="path to basalt calibration.json for camera calibration", required=True)
    try:
        args = parser.parse_args()
    except ArgumentError as e:
        print(e.message)
        parser.print_help()
        return 1

    stitcher = Stitcher(args.calibration)
    stitcher.resolution(3200, 1600)
    stitcher.setDepth(1)

    num_cams = len(args.inputs)
    caps = list()

    for input in args.inputs:
        cap = cv.VideoCapture(input)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video {input}")
        caps.append(cap)

    for i in range(num_cams):
        cv.namedWindow(f"Input{i}", cv.WINDOW_NORMAL)
    cv.namedWindow("Output", cv.WINDOW_NORMAL)
    cv.createTrackbar("Resolution X (px)", "Output", 0, 4000, lambda x: stitcher.width(x if x > 0 else 1))
    cv.createTrackbar("Resolution Y (px)", "Output", 0, 4000, lambda x: stitcher.height(x if x > 0 else 1))
    cv.createTrackbar("FOV X (deg)", "Output", 0, 360, lambda x: stitcher.fovX(x / 180 * pi))
    cv.createTrackbar("FOV Y (deg)", "Output", 0, 180, lambda x: stitcher.fovY(x / 180 * pi))
    cv.createTrackbar("Depth (cm)", "Output", 0, 1000, lambda x: stitcher.setDepth(x / 100 if x > 0 else 1 / 1000))
    cv.createTrackbar("Vignette Threshold (%)", "Output", 0, 100, lambda x: stitcher.vignetteThreshold(x / 100))
    cv.createTrackbar("Rotation X (deg)", "Output", 0, 360, lambda x: stitcher.transform(modifyRotation(stitcher.transform(), x = x)))
    cv.createTrackbar("Rotation Y (deg)", "Output", 0, 360, lambda x: stitcher.transform(modifyRotation(stitcher.transform(), y = x)))
    cv.createTrackbar("Rotation Z (deg)", "Output", 0, 360, lambda x: stitcher.transform(modifyRotation(stitcher.transform(), z = x)))

    ret, imgs = readInputs(caps)
    while ret:
        for i, img in enumerate(imgs):
            cv.imshow(f"Input{i}", img)

        pano = stitcher.stitch(imgs)

        cv.imshow("Output", pano)
        if cv.waitKey(1) == 27:
            break

        ret, imgs = readInputs(caps)

if __name__ == "__main__":
    main()
