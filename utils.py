import os
import yaml
import torch as th
from omnigibson.utils.control_utils import orientation_error
from omnigibson.controllers import InverseKinematicsController
import omnigibson.utils.transform_utils as T

def get_config(config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config



def generate_empty_action(self):
    """
    Generate a no-op action that will keep the robot still but aim to move the arms to the saved pose targets, if possible

    Returns:
        th.tensor or None: Action array for one step for the robot to do nothing
    """
    action = th.zeros(self.robot.action_dim)
    for name, controller in self.robot._controllers.items():
        # if desired arm targets are available, generate an action that moves the arms to the saved pose targets
        if name in self._arm_targets:
            if isinstance(controller, InverseKinematicsController):
                arm = name.replace("arm_", "")
                target_pos, target_orn_axisangle = self._arm_targets[name]
                current_pos, current_orn = self._get_pose_in_robot_frame(
                    (self.robot.get_eef_position(arm), self.robot.get_eef_orientation(arm))
                )
                delta_pos = target_pos - current_pos
                if controller.mode == "pose_delta_ori":
                    delta_orn = orientation_error(
                        T.quat2mat(T.axisangle2quat(target_orn_axisangle)), T.quat2mat(current_orn)
                    )
                    partial_action = th.cat((delta_pos, delta_orn))
                elif controller.mode in "pose_absolute_ori":
                    partial_action = th.cat((delta_pos, target_orn_axisangle))
                elif controller.mode == "absolute_pose":
                    partial_action = th.cat((target_pos, target_orn_axisangle))
                else:
                    raise ValueError("Unexpected IK control mode")
            else:
                target_joint_pos = self._arm_targets[name]
                current_joint_pos = self.robot.get_joint_positions()[self._manipulation_control_idx]
                if controller.use_delta_commands:
                    partial_action = target_joint_pos - current_joint_pos
                else:
                    partial_action = target_joint_pos
        else:
            partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
        action_idx = self.robot.controller_action_idx[name]
        action[action_idx] = partial_action
    return action