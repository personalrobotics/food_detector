
def print_usage(err_msg):
    print(err_msg)
    print('Usage:')
    print('\t./detection_w_projection.py <config_filename (e.g. herb)>\n')


def run_detection():
    global config
    config = load_configs()
    if config is None:
        return

    if config.use_cuda == 'true':
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    rospy.init_node(config.node_title)
    rcnn_projection = FoodDetection(
        title=config.node_title,
        use_cuda=(config.use_cuda == 'true'),
        use_spnet=False,
        use_model1=False)

    try:
        pub_pose = rospy.Publisher(
            '{}/marker_array'.format(config.node_title),
            MarkerArray,
            queue_size=1)

        rate = rospy.Rate(config.frequency)

        while not rospy.is_shutdown():
            update_timestamp_str = 'update: {}'.format(rospy.get_time())
            rst = rcnn_projection.detect_objects()

            poses = list()
            pose = Marker()
            pose.header.frame_id = config.camera_tf
            pose.header.stamp = rospy.Time.now()
            pose.id = 0
            pose.ns = 'food_item'
            pose.type = Marker.CUBE
            pose.action = Marker.DELETEALL
            poses.append(pose)

            item_dict = dict()
            if rst is not None:
                for item in rst:
                    if item[2] in item_dict:
                        item_dict[item[2]] += 1
                    else:
                        item_dict[item[2]] = 1

                    obj_info = dict()
                    obj_info['uid'] = '{}_{}'.format(
                        item[2], item_dict[item[2]])

                    pose = Marker()
                    pose.header.frame_id = conf.camera_tf
                    pose.header.stamp = rospy.Time.now()
                    pose.id = item[3] * 1000 + item[5]
                    pose.ns = 'food_item'
                    pose.text = json.dumps(obj_info)
                    pose.type = Marker.CYLINDER
                    pose.pose.position.x = item[1][0]
                    pose.pose.position.y = item[1][1]
                    pose.pose.position.z = item[1][2]
                    pose.pose.orientation.x = item[0][0]
                    pose.pose.orientation.y = item[0][1]
                    pose.pose.orientation.z = item[0][2]
                    pose.pose.orientation.w = item[0][3]
                    pose.scale.x = 0.01
                    pose.scale.y = 0.01
                    pose.scale.z = 0.04
                    pose.color.a = 0.5
                    pose.color.r = 1.0
                    pose.color.g = 0.5
                    pose.color.b = 0.1
                    pose.lifetime = rospy.Duration(0)
                    poses.append(pose)

            pub_pose.publish(poses)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    args = sys.argv

    config_filename = None
    if len(args) == 2:
        config_filename = args[1]
    else:
        ros_param_name = '/pose_estimator/config_filename'
        if rospy.has_param(ros_param_name):
            config_filename = rospy.get_param(ros_param_name)

    if config_filename is None:
        print_usage('Invalid arguments')
        exit(0)

    if config_filename.startswith('ada'):
        from robot_conf import ada as conf
    elif config_filename.startswith('herb'):
        from robot_conf import herb as conf
    else:
        print_usage('Invalid arguments')
        exit(0)

    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    run_detection()
