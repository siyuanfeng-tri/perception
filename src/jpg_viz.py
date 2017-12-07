#!/usr/bin/env python

# export PYTHONPATH=/home/user/drake_bazel_build/lib/python2.7/site-packages

import bot_core
import robotlocomotion
import cv2
import lcm
import numpy as np
import time

_VIEW_UPDATE_RATE = 30. # Hz

class ImageViewer(object):
    def __init__(self, channel_name):
        self._subscribed_msg = None
        self._lcm = lcm.LCM()
        self._lcm.subscribe(channel_name, self.callback)

    def callback(self, channel, data):
        self._subscribed_msg = bot_core.raw_t.decode(data)

    def spin_once(self):
        self._lcm.handle()

    def get_image(self):
        if self._subscribed_msg is None:
            return None

        raw_bytes = np.array(list((self._subscribed_msg.data)))
        raw_bytes = raw_bytes.view(np.uint8)

        img = cv2.imdecode(raw_bytes, cv2.CV_LOAD_IMAGE_COLOR)
        return img

def main():
    rgb_handler = ImageViewer("RGB_IMG")
    depth_handler = ImageViewer("DEPTH_IMG")

    handlers = [rgb_handler, depth_handler]
    names = ['Color Image', 'Depth Image']

    try:
        while True:
            start_time = time.time()
            for i in range(0, len(handlers)):
                handlers[i].spin_once()

                img = handlers[i].get_image()
                if img is not None:
                    cv2.imshow(names[i], img)
                    cv2.waitKey(2)

            while time.time() - start_time < 1 / _VIEW_UPDATE_RATE:
                time.sleep(0.005)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
