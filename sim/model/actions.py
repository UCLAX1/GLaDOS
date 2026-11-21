import mujoco
import time
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.composites import Sequence
from py_trees import logging as log_tree
from robot_context import RobotContext
import numpy as np

# r theta
def spherical_to_cartesian(spherical_coords):
    r = spherical_coords[0]
    theta = spherical_coords[1]
    phi = spherical_coords[2]
    return r * np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


class MoveArm(Behaviour):
    def __init__(
            self, 
            name, 
            robot_context: RobotContext,
            target_spherical_coords, # r, theta, phi
            target_head_yaw,
            target_head_pitch,
        ):
        super(MoveArm, self).__init__(name)

        self.robot_context = robot_context
        self.target_spherical_coords = target_spherical_coords
        self.target_arm_position_vector = spherical_to_cartesian(target_spherical_coords)
        self.target_head_yaw = target_head_yaw
        self.target_head_pitch = target_head_pitch

        self.start_time = 0
    
    
    def setup(self):
        self.logger.debug(f"MoveArm::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"MoveArm::initialise {self.name}")
        # move arm

        self.robot_context.set_target_pose_and_solve_ik(self.target_arm_position_vector, self.target_head_yaw, self.target_head_pitch)
        self.robot_context.move_head()
        self.robot_context.move_arm()

        self.start_time = time.time()

    def update(self):
        self.logger.debug(f"MoveArm::update {self.name}")
        # TODO: put code here to only return success when the motor reaches the target destination, otherwise return Status.RUNNING
        status = Status.SUCCESS
        if time.time() - self.start_time < 5:
            status = Status.RUNNING
        return status
        # pass

    def terminate(self, new_status):
        self.logger.debug(f"MoveArm::terminate {self.name} to {new_status}")
        exit(0)
        # pass

def make_bt(
    robot_context: RobotContext
    ):
    # root = Sequence(name="sequence", memory=True)
    # action = Action("my action")

    # root.add_children([
    #     action
    # ])

    # return root

    move_arm = MoveArm("move_arm", robot_context, [1.3, np.deg2rad(0), np.deg2rad(148)], np.deg2rad(0), np.deg2rad(0))
    return move_arm