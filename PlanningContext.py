from omnigibson.utils.control_utils import FKSolver
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
import torch as th

class PlanningContext(object):
    """
    A context manager that sets up a robot copy for collision checking in planning.
    """

    def __init__(self, env, robot, robot_copy, robot_copy_type="original"):
        self.env = env
        self.robot = robot
        self.robot_copy = robot_copy
        self.robot_copy_type = robot_copy_type if robot_copy_type in robot_copy.prims.keys() else "original"
        self.disabled_collision_pairs_dict = {}

        # For now, the planning context only works with Fetch and Tiago
        #assert isinstance(self.robot, (Fetch, Tiago)), "PlanningContext only works with Fetch and Tiago."

    def __enter__(self):
        self._assemble_robot_copy()
        self._construct_disabled_collision_pairs()
        return self

    def __exit__(self, *args):
        self._set_prim_pose(
            self.robot_copy.prims[self.robot_copy_type], self.robot_copy.reset_pose[self.robot_copy_type]
        )

    def _assemble_robot_copy(self):
        if False:
            fk_descriptor = "left_fixed"
        else:
            fk_descriptor = (
                "combined" if "combined" in self.robot.robot_arm_descriptor_yamls else self.robot.default_arm
            )
        self.fk_solver = FKSolver(
            robot_description_path=self.robot.robot_arm_descriptor_yamls[fk_descriptor],
            robot_urdf_path=self.robot.urdf_path,
        )

        # TODO: Remove the need for this after refactoring the FK / descriptors / etc.
        arm_links = self.robot.arm_link_names | self.robot.eef_link_names | self.robot.finger_link_names

        if False:
            assert self.arm == "left", "Fixed torso mode only supports left arm!"
            joint_control_idx = self.robot.arm_control_idx["left"]
            joint_pos = self.robot.get_joint_positions()[joint_control_idx]
        else:
            joint_combined_idx = th.cat([self.robot.arm_control_idx[fk_descriptor]])
            joint_pos = self.robot.get_joint_positions()[joint_combined_idx]
        link_poses = self.fk_solver.get_link_poses(joint_pos, arm_links)

        # Assemble robot meshes
        for link_name, meshes in self.robot_copy.meshes[self.robot_copy_type].items():
            for mesh_name, copy_mesh in meshes.items():
                # Skip grasping frame (this is necessary for Tiago, but should be cleaned up in the future)
                if "grasping_frame" in link_name:
                    continue
                # Set poses of meshes relative to the robot to construct the robot
                link_pose = (
                    link_poses[link_name]
                    if link_name in arm_links
                    else self.robot_copy.links_relative_poses[self.robot_copy_type][link_name]
                )
                mesh_copy_pose = T.pose_transform(
                    *link_pose, *self.robot_copy.relative_poses[self.robot_copy_type][link_name][mesh_name]
                )
                self._set_prim_pose(copy_mesh, mesh_copy_pose)

    def _set_prim_pose(self, prim, pose):
        translation = lazy.pxr.Gf.Vec3d(*pose[0].tolist())
        prim.GetAttribute("xformOp:translate").Set(translation)
        orientation = pose[1][[3, 0, 1, 2]]
        prim.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*orientation.tolist()))

    def _construct_disabled_collision_pairs(self):
        robot_meshes_copy = self.robot_copy.meshes[self.robot_copy_type]

        # Filter out collision pairs of meshes part of the same link
        for meshes in robot_meshes_copy.values():
            for mesh in meshes.values():
                self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] = [
                    m.GetPrimPath().pathString for m in meshes.values()
                ]

        # Filter out all self-collisions
        if self.robot_copy_type == "simplified":
            all_meshes = [
                mesh.GetPrimPath().pathString
                for link in robot_meshes_copy.keys()
                for mesh in robot_meshes_copy[link].values()
            ]
            for link in robot_meshes_copy.keys():
                for mesh in robot_meshes_copy[link].values():
                    self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += all_meshes
        # Filter out collision pairs of meshes part of disabled collision pairs
        else:
            for pair in self.robot.disabled_collision_pairs:
                link_1 = pair[0]
                link_2 = pair[1]
                if link_1 in robot_meshes_copy.keys() and link_2 in robot_meshes_copy.keys():
                    for mesh in robot_meshes_copy[link_1].values():
                        self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += [
                            m.GetPrimPath().pathString for m in robot_meshes_copy[link_2].values()
                        ]

                    for mesh in robot_meshes_copy[link_2].values():
                        self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += [
                            m.GetPrimPath().pathString for m in robot_meshes_copy[link_1].values()
                        ]

        # Filter out colliders all robot copy meshes should ignore
        disabled_colliders = []

        # Disable original robot colliders so copy can't collide with it
        disabled_colliders += [link.prim_path for link in self.robot.links.values()]
        filter_categories = ["floors", "carpet"]
        for obj in self.env.scene.objects:
            if obj.category in filter_categories:
                disabled_colliders += [link.prim_path for link in obj.links.values()]

        # Disable object in hand
        obj_in_hand = self.robot._ag_obj_in_hand[self.robot.default_arm]
        if obj_in_hand is not None:
            disabled_colliders += [link.prim_path for link in obj_in_hand.links.values()]

        for colliders in self.disabled_collision_pairs_dict.values():
            colliders += disabled_colliders