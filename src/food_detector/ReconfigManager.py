#!/usr/bin/env python

import rospy
import numpy as np
import sys
import os

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
        self.threshold = 1
        self.pushVectors = {'left':[1,0,0], 'up':[0,-1,0], 'right':[-1,0,0], 'down':[0,1,0]}
        self.reset()
        self.timer = my_timer()
        self.allow_push_img = None

    def reset(self):
        '''
            reset all variables used in reconfig food item.
        '''
        print("## ----------------------Reset and Start---------------------- ##")
        self.state = StateMachine.FREE
        self.lock = True
        ## store 'raw', 'left', 'up', 'right', 'down' image
        self.img_buffer = {}
        ## store 'raw', 'left', 'up', 'right', 'down' marker_array corresponding to img_buffer returned by SPANet
        self.marker_array_buffer = {}
        self.final_marker_array = MarkerArray()
        self.final_score = 0
        self.final_action = None

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

            print('out = ' + str(yaml.load(self.img_buffer['raw'].header.frame_id)))

            self.spanet_pub.publish(self.img_buffer['raw'])
            self.state = StateMachine.ONE

            # for restart due to time out
            self.last_call_from_spanet = time.time()
        else:
            time_elapsed_for_spanet = time.time() - self.last_call_from_spanet
            if (time_elapsed_for_spanet > 3):
                print("time out for one, reset")
                # os._exit(1)
                self.reset()
            else:
                pass
                # print('0. Ignore camera img, not FREE, it\'s {}'.format(str(self.state)))

    def spanet_callback(self, marker_array):

        # print("************************************************************call time count************************************************************")

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

            ## this callback requires that the length of marker_array returned by spanet must be 1: len(marker_array) = 1
        '''

        self.last_call_from_spanet = time.time()

        if self.state == StateMachine.ONE:

            print('1. StateMachine.ONE: ')
            self.marker_array_buffer['raw'] = deepcopy(marker_array)
            best_score = 0
            best_action = None
            push_direction = None
            for i, marker in enumerate(marker_array.markers):
                yaml_node = yaml.load(marker.text)
                score = yaml_node['score']
                push_direction = yaml_node['push_direction']
                if best_score < score:
                    best_score = score
                    best_action = yaml_node['action']
            self.final_score = best_score
            self.final_marker_array = deepcopy(marker_array)

            print('receive = ' + str(push_direction))

            self.allow_push_img = rospy.get_param('/pushingDemo/allow_push_img')
            if best_score > self.threshold or (not self.allow_push_img):
                print("1. No push, " + "best action is {}, best score is {}".format(best_action, str(best_score)))

                self.state = StateMachine.READY
                print('1. find good enough action with raw image, change to {}.'.format(str(self.state)))

            else:
                print("1. Need to reconfig the food item via pushing action")
                self.generatePushedImage(generator='RF')
                self.state = StateMachine.PUSHING

                self.allow_push_img = False
                rospy.set_param('/pushingDemo/allow_push_img',  self.allow_push_img)

                self.wait_for_state_change()

            return

        if self.state == StateMachine.PUSHING and self.lock:
            # print("2. StateMachine.PUSHING:")
            # print("pushing" + marker_array.markers[0].text)
            push_direction = None
            best_score = 0
            best_action = None
            for marker in marker_array.markers:
                yaml_node = yaml.load(marker.text)
                push_direction = yaml_node['push_direction']
                if best_score < yaml_node['score']:
                    best_score = yaml_node['score']
                    best_action = yaml_node['action']

            if self.final_score <= best_score:
                self.final_score = best_score
                self.final_marker_array = deepcopy(marker_array)

            if push_direction is not None:
                # print('2. push_direction = ', push_direction)
                print('receive = ' + str(push_direction))
                self.marker_array_buffer[yaml_node['push_direction']] = deepcopy(marker_array)
            else:
                print('Here None receive = ' + str(push_direction))
            # print('receive in mesh = ' + marker_array.markers[0].mesh_resource)
            print("2. {} push, best action is {}, best score is {}".format(str(push_direction), best_action, str(best_score)))
            print("-------------------------------------------------")

            self.lock = False
            return

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
                # print(self.img_buffer[key].header.frame_id)
        if generator == 'VideoPrediction':
            for key in self.pushVectors.keys():
                self.img_buffer[key] = deepcopy(self.img_buffer['raw'])

    def generate(self, push_direction, push_vec):
        pushed_img = deepcopy(self.img_buffer['raw'])
        info_map = yaml.load(pushed_img.header.frame_id)
        info_map['push_vec'] = push_vec
        info_map['push_direction'] = push_direction
        pushed_img.header.frame_id = yaml.dump(info_map)
        return pushed_img

    def wait_for_spanet(self):
        # print('wait_for_spanet processing PushedImage')
        while(not rospy.is_shutdown() and self.lock):
            pass
        # print("lock status: " + str(self.lock))

    def wait_for_state_change(self):
        '''
            sleep for 0.1s to wait until state change finish
        '''
        rospy.sleep(0.1)

    def spin(self):
        while not rospy.is_shutdown():
            if self.state == StateMachine.PUSHING:
                for key in self.pushVectors.keys():
                    print('out = ' + yaml.load(self.img_buffer[key].header.frame_id)['push_direction'])
                    self.lock = True
                    self.spanet_pub.publish(self.img_buffer[key])
                    # rospy.wait_for_message is not suitable here
                    # rospy.wait_for_message(self.spanet_out_topic, MarkerArray)
                    self.wait_for_spanet()

                self.state = StateMachine.READY

            if self.state == StateMachine.READY:
                print('5. READY to pub final marker_array with best action')
                if len(self.final_marker_array.markers) > 0:
                    self.final_marker_array.markers[0].header.seq = self.timer.seq
                    self.final_pub.publish(self.final_marker_array)
                    print('5. finish publishing final_marker_array, change to {}'.format('StateMachine.FREE'))
                    self.reset()
                    self.timer.tic_tok()
                else:
                    print("5. self.final_marker_array is None, didn't find a good enough potential action")
                    self.reset()

if __name__ == "__main__":
    node_name = 'ReconfigManager'
    rospy.init_node(node_name)
    rm = ReconfigManager(node_name=node_name)

    try:
        rm.spin()

    except rospy.ROSInterruptException:
        pass

