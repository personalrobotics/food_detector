#!/usr/bin/env python

import rospy
import numpy as np

from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import yaml

from repub_manager import RepubManager

from enum import Enum
from copy import deepcopy
import time 

class StateMachine(Enum):
    '''
        Helper Enum 
    '''
    FREE = 0
    ONE = 1
    PUSHING = 2
    READY = 5

class my_timer(object):
    '''
        Helper class, print the time elapsed between two continuous calling method `tik tok`
    '''

    def __init__(self):
        # print("timer init")
        self.t = time.time()
        self.seq = 0
        print("No.{}".format(str(self.seq)))

    def tic_tok(self):
        elapsed = time.time() - self.t
        self.t = time.time()
        self.seq += 1
        print("No.{}, time elapsed = {}".format(str(self.seq), str(elapsed)))

class ReconfigManager(RepubManager):
    def __init__(self, node_name = 'ReconfigManager'):
        RepubManager.__init__(self, node_name)
        self.threshold = 0.1
        # self.pushDirections = ['left', 'up', 'right', 'down']
        self.pushVectors = {'left':[1,0,0], 'up':[0,-1,0], 'right':[-1,0,0], 'down':[0,1,0]}
        # self.state = StateMachine.FREE 
        self.reset()
        self.timer = my_timer()

    def reset(self):
        '''
            reset all variables used in reconfig food item.
        '''
        print("## ----------------------Reset and Start---------------------- ##")        
        self.state = StateMachine.FREE 
        ## store 'raw', 'left', 'up', 'right', 'down' image
        self.img_buffer = {} 
        ## store 'raw', 'left', 'up', 'right', 'down' marker_array corresponding to img_buffer returned by SPANet
        self.marker_array_buffer = {} 
        self.final_marker_array = MarkerArray()
        self.final_score = 0

    def camera_callback(self, img):
        '''
            @param img: sensor_msgs.msg.CompressedImage
        '''
        if self.state == StateMachine.FREE:
            print('0. get img message from camera')
            
            ## add pushing info into info_map stored in CompressedImage.header.frame_id
            info_map = dict(frame_id=img.header.frame_id, push_direction=None, push_vec=[0,0,0])
            self.img_buffer['raw'] = deepcopy(img)
            self.img_buffer['raw'].header.frame_id = yaml.dump(info_map)
            # print("debug header:", self.img_buffer['raw'].header)
            self.spanet_pub.publish(self.img_buffer['raw'])
            self.state = StateMachine.ONE
            print('1. finish publishing raw img to SPANet, change to StateMachine.ONE')
        else: 
            print('0. get img message from camera, but state is not FREE, it\'s {}'.format(str(self.state)))

    def spanet_callback(self, marker_array):
        ## assuming it's all scooping action, no skewering.
        # print('1. {}, get marker_array msg from spanet'.format(self.spanet_callback.__name__))
        ## TODO:
        '''
            add something that can interact with SPANet:
            when the highest score in marker_array is less than 'threshold', 
            then call pushedImageGenerator() method that can return
            #num of pushed image and its corresponding pushing vector in image size,
            then refeed it back to SPANet to get another #num of Marker_Array.
            then pick the highest score one publish it to topic '/food_detector/marker_array'
        '''

        # yaml.dump(item.info_map)
        ## this callback requires that the length of marker_array returned by spanet must be 1: len(marker_array) = 1

        if self.state == StateMachine.ONE:
            print('1. processing marker_array from spanet')
            # print(marker_array.markers[0].header)
            ## add pushing info into info_map stored in marker.text
            self.marker_array_buffer['raw'] = deepcopy(marker_array)
            best_score = 0
            best_action = None
            for i, marker in enumerate(marker_array.markers):
                # print('it\'s NO. {} marker'.format(str(i)))
                yaml_node = yaml.load(marker.text)
                # print(yaml_node)
                for i, score in enumerate(yaml_node['scores']):
                    if best_score < score:
                        best_score = score
                        best_action = yaml_node['actions'][i]
            self.final_score = best_score

            if best_score > self.threshold:
                print("can do it without reconfig the food item" + 
                      "best action is {}, best score is {}".format(best_action, str(best_score)))
                self.final_marker_array = deepcopy(marker_array)
                self.state = StateMachine.READY
                print('1. find good enough action with raw image, change to {}'.format(str(self.state)))

            else:
                print("Need to reconfig the food item via pushing action")
                self.generatePushedImage(generator='RF')
                self.state = StateMachine.PUSHING
                self.wait_for_state_change()

        if self.state == StateMachine.PUSHING and self.lock:
            # print("debug StateMachine.PUSHING:", marker_array)
            push_direction = None
            best_score = 0
            best_action = None
            for i, marker in enumerate(marker_array.markers):
                yaml_node = yaml.load(marker.text)
                # print(yaml_node)
                push_direction = yaml_node['push_direction']
                if best_score < max(yaml_node['scores']):
                    best_score = max(yaml_node['scores'])
                    best_action = yaml_node['actions'][np.argmax(yaml_node['scores'])]

            # print("Reconfig the food item " + 
            #           "best action is {}, best score is {}".format(best_action, str(best_score)))

            if self.final_score < best_score:
                self.final_marker_array = deepcopy(marker_array)

            if push_direction is not None:
                self.marker_array_buffer[yaml_node['push_direction']] = deepcopy(marker_array)

            self.lock = False

    def generatePushedImage(self, generator='RF'):
        '''
            We have two options for PushedImage Generator:
            1. Random Forest
            2. Video Prediction
            ## now just return 'raw' image
        '''
        if generator == 'RF':
            for key in self.pushVectors.keys():
                self.img_buffer[key] = self.generate(key, self.pushVectors[key])
        if generator == 'VideoPrediction':
            for key in self.pushVectors.keys():
                self.img_buffer[key] = deepcopy(self.img_buffer['raw'])

    def generate(self, push_direction, push_vec):
        pushed_img = deepcopy(self.img_buffer['raw'])
        info_map = yaml.load(pushed_img.header.frame_id)
        info_map['push_vec'] = push_vec
        info_map['push_direction'] = push_direction
        pushed_img.header.frame_id = yaml.dump(info_map) 
        # print(pushed_img.header)
        return pushed_img

    def wait_for_spanet(self):
        # print('wait_for_spanet processing PushedImage')
        while(self.lock):
            pass
        # print("lock status: " + str(self.lock))

    def wait_for_state_change(self):
        '''
            sleep for 0.1s to wait until state change finish
        '''
        rospy.sleep(0.01)

    def spin(self):
        while not rospy.is_shutdown():
            if self.state == StateMachine.PUSHING:
                for key in self.pushVectors.keys():
                    # print(self.img_buffer[key].header.frame_id)
                    self.spanet_pub.publish(self.img_buffer[key])
                    self.lock = True
                    rospy.wait_for_message(self.spanet_out_topic, MarkerArray)
                    # self.wait_for_spanet()
                self.state = StateMachine.READY

            if self.state == StateMachine.READY:
                print('5. READY to pub final marker_array with best action')
                # print("send, {} time".format(str(cnt)))
                if len(self.final_marker_array.markers) > 0:
                    # seq = self.final_marker_array.markers[0].header.seq
                    # print("seq = ", seq)
                    # print(self.final_marker_array)
                    self.final_marker_array.markers[0].header.seq = self.timer.seq
                    self.final_pub.publish(self.final_marker_array)
                    print('5.finish publishing final_marker_array, change to {}'.format('StateMachine.FREE'))
                    self.reset()
                    self.timer.tic_tok()
                else: 
                    print("self.final_marker_array is None, didn't find a good enough potential action")            
                    self.reset()

if __name__ == "__main__":
    node_name = 'ReconfigManager'
    rospy.init_node(node_name)
    rm = ReconfigManager(node_name=node_name)

    try: 
        rm.spin()

    except rospy.ROSInterruptException:
        pass

