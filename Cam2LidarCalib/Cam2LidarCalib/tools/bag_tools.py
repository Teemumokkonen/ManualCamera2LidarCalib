from rosbags.highlevel import AnyReader
from collections import deque
from pathlib import Path
from cv_bridge import CvBridge
import numpy as np
import struct

from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2

class BagReader():
    def __init__(self, filepath, img_topic, cloud_topic) -> None:
        self.bag_file = filepath
        self.img_topic = img_topic
        self.cloud_topic = cloud_topic

        self.image_queue = deque()
        self.cloud_queue = deque()
        self.bridge = CvBridge()

        with AnyReader([Path(filepath)]) as reader:
            camera_connections = [x for x in reader.connections if x.topic in self.img_topic]
            lidar_connections = [x for x in reader.connections if x.topic in self.cloud_topic]

            for connection, t, rawdata in reader.messages(connections=camera_connections + lidar_connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                timestamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

                if connection.topic in self.img_topic:
                    self.image_queue.append((timestamp, msg))
                elif connection.topic in self.cloud_topic:
                    self.cloud_queue.append((timestamp, msg))

        print("data loaded from bag")
        
    def convert_rosbag_cloud_to_sensor_msgs(self, cloud_msg):
        # Manually construct a PointCloud2 message
        msg = PointCloud2()
        print(type(cloud_msg.header))
        msg.header = cloud_msg.header
        msg.height = cloud_msg.height
        msg.width = cloud_msg.width
        msg.fields = cloud_msg.fields
        msg.is_bigendian = cloud_msg.is_bigendian
        msg.point_step = cloud_msg.point_step
        msg.row_step = cloud_msg.row_step
        msg.is_dense = cloud_msg.is_dense
        msg.data = cloud_msg.data
        return msg

    def pointcloud2_to_array(self, cloud_msg: PointCloud2):
        print(type(cloud_msg))

        point_list = []
        for point in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            point_list.append([point[0], point[1], point[2], point[3]])

        # Convert the list to a numpy array for easier handling
        points_array = np.array(point_list, dtype=np.float32)
        return points_array


    def read_rosbag_pointcloud(self, cloud_msg):
        # Extract important fields from the cloud message
        data = cloud_msg.data
        point_step = cloud_msg.point_step
        width = cloud_msg.width
        height = cloud_msg.height
        fields = {f.name: (f.offset, f.datatype) for f in cloud_msg.fields}
        
        # Define a format string based on the fields you want
        fmt = '<'  # Little endian
        fmt += 'fff'  # x, y, z are floats
        if 'i' in fields:
            fmt += 'f'  # intensity is also a float
        point_size = struct.calcsize(fmt)

        # Ensure format string and point step match
        if point_size != point_step:
            raise ValueError("Mismatch between calculated point size and point step.")

        # Parse points
        points = []
        for i in range(height * width):
            offset = i * point_step
            x, y, z, intensity = struct.unpack_from(fmt, data, offset=offset)
            points.append((x, y, z, intensity))
        return np.array(points)


    def get_frame(self):
        img = self.bridge.imgmsg_to_cv2(self.image_queue[0][1])
        cloud = self.read_rosbag_pointcloud(self.cloud_queue[0][1])
        return (img, cloud)
    

    def pop_frame(self):
        self.image_queue.popleft()
        self.cloud_queue.popleft()

    def has_frames(self):
        if len(self.image_queue) > 0 and len(self.cloud_queue) > 0:
            return True

        else:
            return False
             