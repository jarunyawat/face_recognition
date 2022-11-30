#!/usr/bin/python3

import py_trees
from std_srvs.srv import Empty
from std_msgs.msg import Int8

class PeopleDetectBehavior(py_trees.behaviour.Behaviour):
    """
    Node Name :
        * node name *
    Client :
        * /ability_name/enable *
        call service ... in ... server node
    Subscription :
        *...*
    ...
    
    """
    print("init_work1")
    def __init__(self):
        super(PeopleDetectBehavior,self).__init__()
        self.param_status = Int8()
        self.param_status.data = 0

        self.node = None
        self.topic_subscription = None
        self.enable_client = None

    def setup(self,**kwargs):
        self.node = kwargs['node']
        self.enable_client = self.node.create_client(Empty,'/people_detection/enable')
        self.topic_subscription = self.node.create_subscription(Int8,'/people_detection/status',self.subscription_callback, qos_profile = 10)                                                                                                                    

    def initialise(self):
        pass

    def send_enable_request(self):
        req = Empty.Request()
        self.future = self.enable_client.call_async(req)

    def subscription_callback(self,msg):
        self.param_status = msg

    def update(self) -> py_trees.common.Status:

        if self.param_status.data == 2: #change to be your condition
            #ทำงานเสร็จค่อย 1
            return py_trees.common.Status.SUCCESS

        elif self.param_status.data == 0: #change to be your condition
            self.send_enable_request()

            return py_trees.common.Status.RUNNING
        
        elif self.param_status.data == 1: #change to be your condition

            return py_trees.common.Status.RUNNING

        elif self.param_status.data == -1: #change to be your condition
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass
    