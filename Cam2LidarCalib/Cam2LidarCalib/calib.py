import argparse
from tools.bag_tools import BagReader
from tools.visualizer import PangolinViz
from tools.rect_projection import PinHoleProjection
class CalibrationTool():
    def __init__(self, args) -> None:
        self.args = args
        bag_queue = BagReader(args.bagfile, args.image_topic, args.pointcloud_topic)
        projection = PinHoleProjection()
        self.visualizer = PangolinViz(bag_queue, projection)

    def run(self):
        self.visualizer.run()


def main():
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