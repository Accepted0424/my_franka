import random
from utils import get_config, generate_empty_action
from PlanningContext import PlanningContext
import omnigibson as og
from omnigibson.utils.transform_utils import relative_pose_transform
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky, get_grasp_position_for_open
from omnigibson.utils.control_utils import IKSolver
from omnigibson.utils.motion_planning_utils import plan_arm_motion
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
import torch as th
import numpy as np
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T

class RobotCopy:
    """A data structure for storing information about a robot copy, used for collision checking in planning."""

    def __init__(self):
        self.prims = {}
        self.meshes = {}
        self.relative_poses = {}
        self.links_relative_poses = {}
        self.reset_pose = {
            "original": (th.tensor([0, 0, -5.0], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32)),
            "simplified": (th.tensor([5, 0, -5.0], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32)),
        }

class Main:
    def __init__(self):
        global_config = get_config(config_path='config.yaml')
        self.config = global_config['main']
        self.env_config = global_config['env']
        self.env_config['scene']['scene_file'] = 'og_scene_file_pen.json'
        self.max_steps = -1
        self.arm_targets = {}
        self.set_env()
        self.grasp_obj = self.get_obj_by_name('pen_1')
        self.generate_grasp_action(self.grasp_obj)

    def get_obj_by_name(self, name):
        for obj in self.og_env.scene.objects:
            if obj.name == name:
                return obj
        return None
    
    def set_env(self):
        self.og_env = og.Environment(dict(scene=self.env_config['scene'], robots=[self.env_config['robot']['robot_config']], env=self.env_config['og_sim']))
        self.robot = self.og_env.robots[0]
        self.robot_copy = self.load_robot_copy()
        # Allow user to move camera more easily
        og.sim.enable_viewer_camera_teleoperation()
        
    def load_robot_copy(self):
        """Loads a copy of the robot that can be manipulated into arbitrary configurations for collision checking in planning."""
        robot_copy = RobotCopy()

        robots_to_copy = {"original": {"robot": self.robot, "copy_path": self.robot.prim_path + "_copy"}}

        for robot_type, rc in robots_to_copy.items():
            copy_robot = None
            copy_robot_meshes = {}
            copy_robot_meshes_relative_poses = {}
            copy_robot_links_relative_poses = {}

            # Create prim under which robot meshes are nested and set position
            lazy.omni.usd.commands.CreatePrimCommand("Xform", rc["copy_path"]).do()
            copy_robot = lazy.omni.isaac.core.utils.prims.get_prim_at_path(rc["copy_path"])
            reset_pose = robot_copy.reset_pose[robot_type]
            translation = lazy.pxr.Gf.Vec3d(*reset_pose[0].tolist())
            copy_robot.GetAttribute("xformOp:translate").Set(translation)
            orientation = reset_pose[1][[3, 0, 1, 2]]
            copy_robot.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*orientation.tolist()))

            robot_to_copy = None
            if robot_type == "simplified":
                robot_to_copy = rc["robot"]
                self.env.scene.add_object(robot_to_copy)
            else:
                robot_to_copy = rc["robot"]

            # Copy robot meshes
            for link in robot_to_copy.links.values():
                link_name = link.prim_path.split("/")[-1]
                for mesh_name, mesh in link.collision_meshes.items():
                    split_path = mesh.prim_path.split("/")
                    # Do not copy grasping frame (this is necessary for Tiago, but should be cleaned up in the future)
                    if "grasping_frame" in link_name:
                        continue

                    copy_mesh_path = rc["copy_path"] + "/" + link_name
                    copy_mesh_path += f"_{split_path[-1]}" if split_path[-1] != "collisions" else ""
                    lazy.omni.usd.commands.CopyPrimCommand(mesh.prim_path, path_to=copy_mesh_path).do()
                    copy_mesh = lazy.omni.isaac.core.utils.prims.get_prim_at_path(copy_mesh_path)
                    relative_pose = T.relative_pose_transform(
                        *mesh.get_position_orientation(), *link.get_position_orientation()
                    )
                    relative_pose = (relative_pose[0], th.tensor([0, 0, 0, 1]))
                    if link_name not in copy_robot_meshes.keys():
                        copy_robot_meshes[link_name] = {mesh_name: copy_mesh}
                        copy_robot_meshes_relative_poses[link_name] = {mesh_name: relative_pose}
                    else:
                        copy_robot_meshes[link_name][mesh_name] = copy_mesh
                        copy_robot_meshes_relative_poses[link_name][mesh_name] = relative_pose

                copy_robot_links_relative_poses[link_name] = T.relative_pose_transform(
                    *link.get_position_orientation(), *self.robot.get_position_orientation()
                )

            if robot_type == "simplified":
                self.env.scene.remove_object(robot_to_copy)

            robot_copy.prims[robot_type] = copy_robot
            robot_copy.meshes[robot_type] = copy_robot_meshes
            robot_copy.relative_poses[robot_type] = copy_robot_meshes_relative_poses
            robot_copy.links_relative_poses[robot_type] = copy_robot_links_relative_poses

        og.sim.step()
        return robot_copy

    def generate_grasp_action(self, obj):
        """
        action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
        """

        # Get the grasp pose and direction for the object
        grasp_poses = get_grasp_poses_for_object_sticky(obj)
        grasp_pose, object_direction = random.choice(grasp_poses)
        
        # Prepare data for the approach later.
        approach_pos = grasp_pose[0] + object_direction * 0.2 # grsap approach distance is 0.2m
        approach_pose = (approach_pos, grasp_pose[1])

        # move hand to the approach pose
        controller_config = self.robot._controller_config['arm_' + self.robot.default_arm]
        if controller_config["name"] == "InverseKinematicsController":
            # undone
            return None
        else:
            # convert cartesian to joint space
            body_pose = self.robot.get_position_orientation()
            relative_target_pose = relative_pose_transform(*grasp_pose, *body_pose)
            
            # IK-Solver cartesian to joint space
            manipulation_descriptor_path = self.robot.robot_arm_descriptor_yamls[self.robot.default_arm]
            manipulation_control_idx = self.robot.arm_control_idx[self.robot.default_arm]
            
            ik_solver = IKSolver(
                robot_description_path=manipulation_descriptor_path,
                robot_urdf_path=self.robot.urdf_path,
                reset_joint_pos=self.robot.reset_joint_pos[manipulation_control_idx],
                eef_name=self.robot.eef_link_names[self.robot.default_arm],
            )

            joint_pos = ik_solver.solve(
                target_pos=relative_target_pose[0],
                target_quat=relative_target_pose[1],
                max_iterations=100,
            )

            with PlanningContext(self.og_env, self.robot, self.robot_copy, "original") as context:
                plan = plan_arm_motion(
                    robot=self.robot,
                    end_conf=joint_pos,
                    context=context,
                    torso_fixed=False,
                )

            assert plan is not None, "Failed to plan to the target pose."
            
            for i, joint_pos in enumerate(plan):
                print(f"Executing arm movement plan step {i + 1}/{len(plan)}")
                # self._move_hand_direct_joint(joint_pos, ignore_failure=True)
                prev_eef_pos = th.zeros(3)

                controller_name = f"arm_{self.robot.default_arm}"
                self.arm_targets[controller_name] = joint_pos

                # max steps is 500
                for i in range(500):
                    current_joint_pos = self.robot.get_joint_positions()[self.robot.arm_joint_indices[self.robot.default_arm]] #default arm control idx
                    diff_joint_pos = joint_pos - current_joint_pos
                    if th.max(th.abs(diff_joint_pos)).item() < 0.01: # 0.01 is the threshold
                        return
                
                # Generate a no-op action that will keep the robot still but aim to move the arms to the saved pose targets, if possible
                action = th.zeros(self.robot.action_dim)
                for name, controller in self.robot._controllers.items():
                    if name in self.arm_targets:
                        # the controller is not "InverseKinematicsController"
                        target_joint_pos = self.arm_targets[name]
                        current_joint_pos = self.robot.get_joint_positions()[self.robot.arm_joint_indices[self.robot.default_arm]]
                        if controller.use_delta_commands:
                            partial_action = target_joint_pos - current_joint_pos
                        else:
                            partial_action = target_joint_pos
                    else:
                        partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
                    action_idx = self.robot.controller_action_idx[name]
                    action[action_idx] = partial_action
                self.env.step(action)
        

if __name__ == '__main__':
    main = Main()
        


        
