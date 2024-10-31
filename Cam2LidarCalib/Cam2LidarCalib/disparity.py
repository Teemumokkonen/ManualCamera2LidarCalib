import argparse
import cv2
from tools.bag_tools import BagReader
from tools.visualizer import PangolinViz
from tools.rect_projection import PinHoleProjection
from tools.depth_estimator import DepthEstimator
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from sensor_msgs_py.point_cloud2 import create_cloud
import rclpy
from rclpy.node import Node
import struct
from std_msgs.msg import Header

class CloudPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'disparity/cloud', 10)

    def create_pointcloud2_msg(self, points, colors, frame_id="kitti_velo"):
        header = Header()
        header.frame_id = "kitti_velo"
        header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Pack the RGB colors as floats
        cloud_data = []
        for (x, y, z), (r, g, b) in zip(points, colors):
            rgb = struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]
            cloud_data.append([x, y, z, rgb])

        # Create the PointCloud2 message
        pointcloud_msg = create_cloud(header, fields=fields, points=cloud_data)
        return pointcloud_msg

    def pub_cloud(self, xyz, rgb):
        msg = PointCloud2()
        header = Header()
        header.frame_id = "kitti_velo"
        header.stamp = self.get_clock().now().to_msg()
        cloud = self.create_pointcloud2_msg(xyz, rgb)
        self.publisher_.publish(cloud)


class CalibrationTool():
    def __init__(self, args) -> None:
        self.args = args
        self.bag_queue = BagReader(args.bagfile, args.image_topic, args.pointcloud_topic)
        self.estimator = DepthEstimator()
        self.ros_node = CloudPublisher()
        #projection = PinHoleProjection()
        #self.visualizer = PangolinViz(bag_queue, projection)

    def run(self):
        #self.visualizer.run()
        
        while True:
            img = self.bag_queue.get_frame()[0]
            height, width, _    = img.shape
            print(width)
            print(height)
            x = np.tile(np.arange(width), height)
            y = np.repeat(np.arange(height), width)[::-1]

            xyz = np.zeros((width * height, 3), dtype=int)
            xyz[:,0] = x
            xyz[:,1] = y
            disparity_img = self.estimator.predict(img)
            #cv2.imshow("disparity map", disparity_img)
            #cv2.imshow("original image", img)
            #cv2.waitKey(0)
            self.bag_queue.pop_frame()
            xyz[:,2] = disparity_img.flatten()
            rgb = img.reshape(-1, img.shape[-1])
            rgb = rgb[:,[2,1,0]]
            rgb = rgb.astype('float')
            self.ros_node.pub_cloud(xyz, rgb)


def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="The bag file")
    parser.add_argument("image_topic", nargs="+", help="Image topic for camera calibration")
    parser.add_argument("pointcloud_topic", nargs="+", help="Point cloud topic for LiDAR calibration")
    parser.add_argument("--start", type=float, default=0, help="Start timestamp")
    parser.add_argument("--end", type=float, default=float('inf'), help="End timestamp")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()
    calib_tool = CalibrationTool(args)
    calib_tool.run()

if __name__ == '__main__':
    main()