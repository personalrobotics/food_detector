#!/usr/bin/env python

"""
Initiates the services `getAction` and `publishLoss`.

TODO:
  * Refactor
    - Move heavy-lifting code to `src/food_detector`
    - Move `srv` to `src/food_detector`
  * Run this script as a node in a launch script
  * Use arguments to define `algo`
"""

import rospy
from conban_spanet.conbanalg import singleUCB

from food_detector.srv import GetAction, PublishLoss, GetActionResponse, PublishLossResponse

SERVER_NAME = 'conban_spanet_server'

# TODO: add options for other CONBANs
algo = singleUCB(N=1, alpha=0.05, gamma=0)

def handle_get_action(req):
    p_t = algo.explore(req.features)

    # Sample Action
    _, K = p_t.shape
    p_t_flat = p_t.reshape((-1,))
    sample_idx = np.random.choice(N*K, p = p_t_flat)
    a_t = sample_idx % K

    return srv.GetActionResponse(a_t, p_t)

def handle_publish_loss(req):
    try:
        algo.learn(req.features, 0, req.a_t, req.loss, req.p_t)
    except:
        return False
    return True

if __name__ == '__main__':
    rospy.init_node(SERVER_NAME)
    rospy.Service('GetAction', GetAction, handle_get_action)
    rospy.Service('PublishLoss', PublishLoss, handle_publish_loss)
    try:
        print('Server running')
        rospy.spin()
    except KeyboardInterrupt:
        pass
    print('Shutting down {}...'.format(SERVER_NAME))
