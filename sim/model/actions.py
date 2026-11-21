import mujoco
import time
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.composites import Sequence
from py_trees import logging as log_tree
from robot_context import RobotContext


class MoveArm(Behaviour):
    def __init__(
            self, 
            name, 
            robot_context: RobotContext,
            target_arm_position_vector,
        ):
        super(Action, self).__init__(name)

        self.arm_solver.set_target_pose(target_arm_position_vector, head_yaw, head_pitch)
        self.target_arm_angles, self.target_head_rotation = self.arm_solver.get_target_angles()

    
    
    def setup(self):
        self.logger.debug(f"MoveArm::setup {self.name}")
        init_sliders()

    def initialise(self):
        self.logger.debug(f"MoveArm::initialise {self.name}")
        # move arm

        base_angle = self.target_arm_angles[0]
        shoulder_angle = self.target_arm_angles[1]
        elbow_angle = self.target_arm_angles[2]

        base_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_motor")
        shoulder_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
        elbow_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_motor")

        data.ctrl[base_motor_id] = base_angle
        data.ctrl[shoulder_motor_id] = shoulder_angle
        data.ctrl[elbow_motor_id] = elbow_angle

        # move head
        euler_angles = self.target_rotation.as_euler("zyx")
        angle_x = euler_angles[2]
        angle_y = euler_angles[1]
        angle_z = euler_angles[0]

        x_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_x")
        y_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_y")
        z_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_z")

        data.ctrl[x_motor_id] = angle_x
        data.ctrl[y_motor_id] = angle_y
        data.ctrl[z_motor_id] = angle_z
        # pass

    def update(self):
        self.logger.debug(f"MoveArm::update {self.name}")
        # TODO: put code here to only return success when the motor reaches the target destination, otherwise return Status.RUNNING
        # status = Status.SUCCESS
        status = Status.RUNNING
        return status
        # pass

    def terminate(self, new_status):
        self.logger.debug(f"MoveArm::terminate {self.name} to {new_status}")
        # pass

def make_bt(
        model, 
        data, 
        arm_solver: ArmSolver, 
    ):
    # root = Sequence(name="sequence", memory=True)
    # action = Action("my action")

    # root.add_children([
    #     action
    # ])

    # return root

    move_arm = MoveArm("move_arm", model, data, arm_solver)
    return move_arm