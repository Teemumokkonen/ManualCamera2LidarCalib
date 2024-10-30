#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from rosbags.highlevel import AnyReader
from collections import deque
import OpenGL.GL as gl
import pangolin
from cv_bridge import CvBridge
import cv2
from time import sleep
import numpy as np
bridge = CvBridge()

def init_pangolin():
    pangolin.CreateWindowAndBind('Main', 1242 * 2, 375)
    gl.glEnable(gl.GL_DEPTH_TEST)
    
    # Set up the camera with correct aspect ratio
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(1242, 375, 500, 420, 621, 187.5, 0.1, 1000),
        pangolin.ModelViewLookAt(1, 1, -1, 0, 0, 0, pangolin.AxisDirection.AxisY))

    # Main display
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 0.5, 1242.0 / 375.0)  # Adjust bounds for splitting window

    # Create image display
    dimg = pangolin.Display("img")
    dimg.SetBounds(0.0, 1.0, 0.5, 1.0, 1242.0 / 375.0)  # Adjust bounds for the image

    dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

    return scam, dcam, dimg

def extract(args):
    # Initialize Pangolin
    scam, dcam, dimg = init_pangolin()

    # Set up texture for the image
    texture = pangolin.GlTexture(1242, 375, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    # Output directory setup
    if not args.output:
        args.output = os.path.dirname(args.bagfile)
    elif not os.path.exists(args.output):
        os.makedirs(args.output)

    image_queue = deque()
    pointcloud_queue = deque()

    with AnyReader([Path(args.bagfile)]) as reader:
        camera_connections = [x for x in reader.connections if x.topic in args.image_topic]
        lidar_connections = [x for x in reader.connections if x.topic in args.pointcloud_topic]

        for connection, t, rawdata in reader.messages(connections=camera_connections + lidar_connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

            # Check if timestamp is within specified range
            if not float(args.start) < timestamp < float(args.end):
                continue

            if connection.topic in args.image_topic:
                image_queue.append((timestamp, msg))
            elif connection.topic in args.pointcloud_topic:
                pointcloud_queue.append((timestamp, msg))

        # Try to synchronize the messages by matching timestamps
        while image_queue and pointcloud_queue:
            print(len(image_queue))

            # Main Pangolin display loop
            while not pangolin.ShouldQuit():
                image_time, image_msg = image_queue[0]
                pointcloud_time, pointcloud_msg = pointcloud_queue[0]
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                gl.glClearColor(0.95, 0.95, 0.95, 1.0)

                # Convert ROS2 image message to OpenCV image in RGB format
                cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
                cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                cv2.imshow('image',cv_image_rgb)
                cv2.waitKey(0)
                cv_image_rgb = cv2.flip(cv_image_rgb, 0) 
                height, width, _ = cv_image_rgb.shape
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                texture.Upload(cv_image_rgb, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

                # Display the image
                dimg.Activate()
                dcam.Activate(scam)
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()

                pangolin.FinishFrame()
                #cv2.waitKey(1)
                print("next frame")
                # Remove processed items from queues
                image_queue.popleft()
                pointcloud_queue.popleft()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="The bag file")
    parser.add_argument("image_topic", nargs="+", help="Image topic for camera calibration")
    parser.add_argument("pointcloud_topic", nargs="+", help="Point cloud topic for LiDAR calibration")
    parser.add_argument("--start", type=float, default=0, help="Start timestamp")
    parser.add_argument("--end", type=float, default=float('inf'), help="End timestamp")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()
    extract(args)
