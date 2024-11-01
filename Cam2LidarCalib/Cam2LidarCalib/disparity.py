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
        self.publisher_ = self.create_publisher(PointCloud2, 'disparity/cloud', 1)
        self.cloud_pub = self.create_publisher(PointCloud2, "velodyne/cloud", 1)

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

    def create_pointcloud2_msg_xyzI(self, points, frame_id="kitti_velo"):
        header = Header()
        header.frame_id = "kitti_velo"
        header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Pack the RGB colors as floats
        cloud_data = []
        for (x, y, z, i) in points:
            #rgb = struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]
            cloud_data.append([x, y, z, i])

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

    def pub_kitti_cloud(self, cloud):
        msg = PointCloud2()
        header = Header()
        header.frame_id = "kitti_velo"
        header.stamp = self.get_clock().now().to_msg()
        cloud = self.create_pointcloud2_msg_xyzI(cloud)
        self.cloud_pub.publish(cloud)


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
            img, cloud = self.bag_queue.get_frame()
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1) 
            height, width, _    = img.shape
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
            XYZ = []
            extrinsics = np.array([[0.0000000,  1.0000000,  0.0000000, 0.27],
                    [0.0000000,  0.0000000,  1.0000000, -0.48],
                    [1.0000000,  0.0000000,  0.0000000, -0.08],
                    [0.0, 0.0, 0.0, 1.0] ])
            for y in range(height):
                for x in range(width):
                    Z = disparity_img[y, x]
                    if  Z > 0:  # skip invalid disparity values
                        #Z = (fx * baseline) / d
                        X = (x - 6.095593e+02) * Z / 7.215377e+02
                        Y = (y - 1.728540e+02) * Z / 7.215377e+02
                        #print(f"x coordinate of the image: {X}")
                        #print(f"y coordinate of the image {Y}")
                        trans = np.linalg.inv(extrinsics) @ np.array([X, Y, Z, 1]).T
                        #print(trans[:3])
                        XYZ.append([trans[0], trans[1], -trans[2]])
            self.ros_node.pub_cloud(XYZ, rgb)
            self.ros_node.pub_kitti_cloud(cloud)
            #cv2.imshow("disparity map", disparity_img)
            #cv2.imshow("original image", img)
            #cv2.waitKey(0)

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