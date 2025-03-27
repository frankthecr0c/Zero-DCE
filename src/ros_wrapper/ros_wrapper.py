#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import torch
import rospy
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from src.model import model
from src.utils import util
import torchvision.transforms as transforms


class ZeroDceRos:

    def __init__(self, config_opt):
        self.opt = config_opt
        self.ros_opt = config_opt["ROS"]
        self.node_name = self.ros_opt["Node"].get('Name', 'zero_dce_enhancer')
        rospy.init_node(self.node_name, anonymous=False)
        self.bridge = CvBridge()
        self.root_path = Path(util.get_parent(os.path.abspath(__file__), levels=2))
        self.weights_path = Path(self.root_path, "snapshots",
                                 self.opt["model"].get('loaded_model', 'Epoch99.pth"'))

        if torch.cuda.is_available():
            with torch.no_grad():
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                self.DCE_net = model.enhance_net_nopool().cuda()
        else:
            rospy.loginfo("No cuda device detected! Working with CPU")
            self.DCE_net = model.enhance_net_nopool()

        self.DCE_net.load_state_dict(torch.load(self.weights_path.__str__()))

        self.pub_topic = self.ros_opt["Node"]["Topics"].get('enhancing_out', "img_enhanced_zeroDCE")
        self.sub_topic = self.ros_opt["Node"]["Topics"].get('enhancing_in', "img_req_enhancer")
        self.img_encoding = self.ros_opt["Image"].get('format_out', 'rgb8')

        self.image_pub = rospy.Publisher(self.pub_topic,  Image, queue_size=1)
        self.image_sub = rospy.Subscriber(self.sub_topic, Image, self._img_callback)

        # Start ros loop
        rospy.spin()

    def _img_callback(self, img_msg):
        with torch.no_grad():
            try:
                cv_image_in = self.bridge.imgmsg_to_cv2(img_msg, self.img_encoding)
                data_lowlight_norm = cv_image_in / 255.0
                data_lowlight = np.transpose(data_lowlight_norm, (2, 0, 1))
                data_lowlight = torch.from_numpy(data_lowlight).float()
                data_lowlight = data_lowlight.cuda().unsqueeze(0)
                _, enhanced_image, _ = self.DCE_net(data_lowlight)
                try:
                    enhanced_np = enhanced_image.cpu().numpy()
                    enhanced_np = enhanced_np[0]
                    enhanced_np = np.transpose(enhanced_np, (1, 2, 0))
                    enhanced_np_uint8 = (enhanced_np * 255).astype(np.uint8)
                    ros_image_out = self.bridge.cv2_to_imgmsg(enhanced_np_uint8, encoding=self.img_encoding)
                    self.image_pub.publish(ros_image_out)
                except CvBridgeError as e:
                    msg = "Error while trying to convert OpenCV image to ROS: {}".format(e)
                    rospy.logerr(msg)

            except CvBridgeError as e:
                msg = "Error while trying to convert ROS image to OpenCV: {}".format(e)
                rospy.logerr(msg)

if __name__ == "__main__":
    root_path = Path(util.get_parent(os.path.abspath(__file__), levels=2))
    config = util.yaml_parser(Path(root_path, "config", "zero_dce.yaml"))

    with torch.no_grad():
        # Create handler
        handler = ZeroDceRos(config_opt=config)


