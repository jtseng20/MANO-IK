import pybullet as p
import numpy as np
from math import sin, cos
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import functools

import os
import io
import time
import pickle
from ik_util import *
import inspect

from matplotlib import pyplot as plt

# Global structure
originColVector = np.array([0,0,0,1])
GLOBAL_HAND_SCALE = 12
PI = np.pi


class IKHand:
    def __init__(self, side, end_effector_list, weights=None, use_bounds=True):
        self.dof_list = [] # all joints
        self.name_to_transform = {}
        self.transform_list = [] # all transforms
        self.target_body_list = [] # target rigid bodies
        self.body_list = [] # rigid bodies
        self.basis = {}
        self.basis_list = []
        self.setup_hand(side)
        
        self.end_effectors = [] # which transforms are end effectors?
        self.targets = [] # what are the target positions?
        self.hand_id = None
        self.make_visual_hand(side)
        
        # bounds for joints
        self.bounds = []
        self.use_bounds = use_bounds
        
        self.create_bounds(side)
        
        # initialize end effectors and targets
        for x in end_effector_list:
            self.add_end_effector(x)
        # match the target bodies to follow the target positions
        self.reset_target_bodies()
        
        self.weights = np.ones(len(end_effector_list))
        if weights is not None:
            self.weights = np.array(weights)
        # transpose weights
        self.weights = self.weights[None,:].T
        
        # what pose am I at / trying to reach? For smooth damping input
        self.current_pose = np.zeros(len(self.dof_list))
        self.target_pose = np.zeros_like(self.current_pose)
        self.currentVelocity = np.zeros_like(self.current_pose)
    
    """
    Designate a transform to be an end effector by its name string
    """
    def add_end_effector(self, s):
        tr = self.name_to_transform[s]
        self.end_effectors.append(tr)
        T = tr.global_transform()
        self.add_target(T[:3, 3])
    
    """
    Add a target for IK
    """
    def add_target(self, target_pos):
        self.targets.append(target_pos)
        sphereRadius = 0.07
        
        # add target indicator
        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        visualId = p.createVisualShape(p.GEOM_SPHERE,
                                       radius=sphereRadius,
                                       rgbaColor=[1.0, 0.0, 0.0, 1.0],
                                       specularColor=[1, 1, 1])
        uid = p.createMultiBody(1.0,
                                colSphereId,
                                visualId, [100.0, 100.0, 100.0], [0.0, 0.0, 0.0, 1.0])
        disable_collisions(uid)
        self.target_body_list.append(uid)
    
    """
    Create joint limits
    """
    def create_bounds(self, side):
        # default: no bounds
        self.bounds = [(-np.inf, np.inf)] * len(self.dof_list)
        # one finger at a time
        if self.use_bounds:
            # first 6 dofs are unconstrained
            self.bounds[:6] = [(-np.inf, np.inf)] * 6
            # dofs 7-9: index knuckle
            # X: +/- 5 degrees (maybe this is too much)
            self.bounds[8] = (-0.01, 0.01)
            # Y: +/- 24 degrees ?
            self.bounds[7] = (-0.34, 0.34)
            # Z: -45-90 degrees
            self.bounds[6] = (-PI / 4, PI / 2)
            # dofs 10, 11: index upper joints constrained to be positive and less than
            # 90 degrees
            self.bounds[9] = (0, PI / 2)
            self.bounds[10] = (0, PI / 4)
            # dofs 12-14: middle knuckle
            # X: +/- 5 degrees (maybe this is too much)
            self.bounds[13] = (-0.01, 0.01)
            # Y: +/- 24 degrees ?
            self.bounds[12] = (-0.34, 0.34)
            # Z: -45-90 degrees
            self.bounds[11] = (-PI / 4, PI / 2)
            # dofs 15, 16: middle upper joints constrained to be positive and less than
            # 90 degrees
            self.bounds[14] = (0, PI / 2)
            self.bounds[15] = (0, PI / 4)
            # dofs 17-19: pinky knuckle
            # X: +/- 5 degrees (maybe this is too much)
            self.bounds[18] = (-0.01, 0.01)
            # Y: +/- 24 degrees ?
            self.bounds[17] = (-0.34, 0.34)
            # Z: -45-90 degrees
            self.bounds[16] = (-PI / 4, PI / 2)
            # dofs 20, 21: pinky upper joints constrained to be positive and less than
            # 90 degrees
            self.bounds[19] = (0, PI / 2)
            self.bounds[20] = (0, PI / 4)
            # dofs 22-24: ring knuckle
            # X: +/- 5 degrees (maybe this is too much)
            self.bounds[23] = (-0.01, 0.01)
            # Y: +/- 24 degrees ?
            self.bounds[22] = (-0.34, 0.34)
            # Z: -45-90 degrees
            self.bounds[21] = (-PI / 4, PI / 2)
            # dofs 25, 26: ring upper joints constrained to be positive and less than
            # 90 degrees
            self.bounds[24] = (0, PI / 2)
            self.bounds[25] = (0, PI / 4)
            
            # dofs 27-29: thumb knuckle
            # X: +/- 5 degrees (maybe this is too much)
            self.bounds[26] = (-0.01, 0.01)
            # Y: +/- 24 degrees ?
            self.bounds[27] = (-0.17, 2.6)
            # Z: -45-90 degrees
            self.bounds[28] = (-0.7, 0.7)
            # dofs 30, 31: thumb upper joints constrained to be positive and less than
            # 90 degrees
            self.bounds[29] = (0, PI / 2)
            self.bounds[30] = (0, PI / 4)
            
            self.bounds = np.array(self.bounds)
        
    def setup_hand(self, side):
        # Reset the data structure
        transform_list = list()
        name_to_transform = dict()
        dof_list = list()
        motion = list()
        with open("joint_distances.p", "rb") as temp_:
            j_d_l, j_d_r = pickle.load(temp_)
        with open("bases.p", "rb") as temp_:
            basis_l, basis_r = pickle.load(temp_)
        basis = basis_l if side == "left" else basis_r
        # Define the hand
        # first define the 6D transformation from world to the base (wrist)
        base = Translation(transform_list, "j_tx", axis="x").add(Translation(transform_list, "j_ty", axis="y")).add(Translation(transform_list, "j_tz", axis="z")).add(Hinge(transform_list, "j_rx", axis="x")).add(Hinge(transform_list, "j_ry", axis="y")).add(Hinge(transform_list, "j_rz", axis="z")).add(Fixed(transform_list, "base"))

        distances = j_d_l if side == "left" else j_d_r
        distances *= GLOBAL_HAND_SCALE
        
        # set self basis to contain the matrix version of each transform
        self.basis = {key:R.from_euler("xyz", val).as_matrix() for key, val in basis.items()}
        # also store basis matrices in a list
        temp_list = ["index", "index", "index", "middle", "middle", "middle", "pinky", "pinky", "pinky", "ring", "ring", "ring", "thumb0", "thumb", "thumb"]
        self.basis_list = [self.basis[k] for k in temp_list]
        # then, add in each fixed joint from wrist to knuckle, and ball at knuckle, then two hinges
        # create a custom basis for each finger too
        for d, s, full_name in zip(['i', 'm','p','r','t'], distances, ["index", "middle", "pinky", "ring", "thumb"]):
            k, j1, j2, tip = s
            # for thumb only, there are two bases: wrist->thumb0->thumb
            if d == "t":
                # thumb starts at thumb0 instead of thumb
                lbx, lby, lbz = basis[full_name + "0"]
                basis_mat = R.from_euler("xyz", basis[full_name+"0"]).as_matrix()
                
                # now, calculate the relative transform between thumb0 and thumb
                thumb0mat = basis_mat
                thumbmat = R.from_euler("xyz", basis[full_name]).as_matrix()
                thumb0_thumb = thumb0mat.T @ thumbmat
                lbx2, lby2, lbz2 = R.from_matrix(thumb0_thumb).as_euler("xyz")
            else:
                # if not thumb, 0 relative transform for all later joints
                lbx, lby, lbz = basis[full_name]
                basis_mat = R.from_euler("xyz", basis[full_name]).as_matrix()
                lbx2, lby2, lbz2 = 0, 0, 0
            
            # every finger has its name as its second basis
            basis_mat2 = R.from_euler("xyz", basis[full_name]).as_matrix()
                
            # transform j1, j2, tip into basis1, then basis2
            j1 = basis_mat.T @ j1
            j2 = basis_mat2.T @ j2
            tip = basis_mat2.T @ tip
            base.add(Fixed(transform_list, "%s_wrist_to_knuckle" % d, tx= k[0], ty = k[1], tz = k[2], rx = lbx, ry = lby, rz = lbz))\
                .add(Hinge(transform_list, "j_%s_knuckle_x" % d, "x", 0.0))\
                .add(Hinge(transform_list, "j_%s_knuckle_y" % d, "y", 0.0))\
                .add(Hinge(transform_list, "j_%s_knuckle_z" % d, "z", 0.0))\
                .add(Fixed(transform_list, "%s_1" % d, tx= j1[0], ty = j1[1], tz = j1[2], rx = lbx2, ry = lby2, rz = lbz2))\
                .add(Hinge(transform_list, "j_%s_1_2" % d , "x", 0.0))\
                .add(Fixed(transform_list, "%s_2" % d, tx= j2[0], ty = j2[1], tz = j2[2]))\
                .add(Hinge(transform_list, "j_%s_2_tip" % d , "x", 0.0))\
                .add(Fixed(transform_list, "%s_tip" % d, tx= tip[0], ty = tip[1], tz = tip[2]))

        # Build auxiliary data structures
        for i_, tr in enumerate(transform_list):
            tr._index = i_
            name_to_transform[tr.name] = tr
            if tr.num_dofs() > 0:
                dof_list.append(tr)
        
        # Build dependent dofs. Each transform knows all of its ancestor transforms, and each active DoF knows all of its ancestor DoFs (for faster jacobians)
        for tr in transform_list:
            tr.dependent_dofs = tr.parent.dependent_dofs.copy() if tr.parent else set()
            tr.dependent_transforms = tr.parent.dependent_transforms.copy() if tr.parent else set()
            tr.dependent_transforms.add(tr)
            if tr.num_dofs() > 0:
                tr.dependent_dofs.add(tr)
        self.transform_list = transform_list
        self.dof_list = dof_list
        self.name_to_transform = name_to_transform
    
    """
    Make all the visual components
    """
    def make_visual_hand(self, side):
        skin_rgba = [0.9, 0.8, 0.6, 1.0]
        white_rgba = [1, 1, 1, 0.9]
        # uncomment if debug joint positions are needed
        """
        Sphere(self.body_list, self.name_to_transform['base'], radius=0.05, color=white_rgba)
        for d in ['i', 'm','r','p','t']:
            Sphere(self.body_list, self.name_to_transform['%s_wrist_to_knuckle' % d], radius=0.05, color=skin_rgba)
            Sphere(self.body_list, self.name_to_transform['%s_1' % d], radius=0.05, color=skin_rgba)
            Sphere(self.body_list, self.name_to_transform['%s_2' % d], radius=0.05, color=skin_rgba)
            Sphere(self.body_list, self.name_to_transform['%s_tip' % d], radius=0.05, color=skin_rgba)
        """
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        # load hand urdf
        hand_path = 'hand_assets/mano_hand_r_tips.urdf' if side == "right" else 'hand_assets/mano_hand_l_tips.urdf'
        hand_id = p.loadURDF(os.path.join(currentdir, hand_path),
                             useFixedBase=1,globalScaling=GLOBAL_HAND_SCALE)
        disable_collisions(hand_id)
        #make_color(hand_id, (0,0,0,0.1))
        self.hand_id = hand_id
            
    """
    Update joint positions to internally stored target. Useful to separate from force_set_joint_positions if user is interested in using smooth_damp instead of instantly resetting to target (not in use currently).
    """
    def set_joint_positions(self):
        # first, smooth_damp towards self.target_pose
        self.current_pose, self.currentVelocity = self.target_pose, None#smooth_damp(self.current_pose, self.target_pose, self.currentVelocity, 0.01, 1, 0.01)
        for q_i, jnt_i in zip(self.target_pose, self.dof_list):
            jnt_i.set_dof(value=q_i)
            
    """
    Reset joint positions to q
    """
    def force_set_joint_positions(self, q):
        for q_i, jnt_i in zip(q, self.dof_list):
            jnt_i.set_dof(value=q_i)
            
    def set_targets(self, targets):
        self.targets = np.array(targets)
    
    """
    Translate the whole hand somewhere. Optionally, move all of the targets along with it.
    """
    def resetBasePositionAndOrientation(self, pos, orn, reset_targets=False, invisible=False):
        current_pos = []
        for pos_, dof_ in zip(pos, self.dof_list[:3]):
            current_pos.append(dof_.get_dof())
            dof_.set_dof(pos_)
        euler_orn = R.from_quat(orn).as_euler("XYZ")
        current_orn = []
        for val_, dof_ in zip(euler_orn, self.dof_list[3:6]):
            current_orn.append(dof_.get_dof())
            dof_.set_dof(val_)
        
        if reset_targets:
            # translate targets to end effectors
            for i, tr in enumerate(self.end_effectors):
                T = tr.global_transform()
                self.targets[i] = T[:3, 3]
              
            # update target bodies
            self.reset_target_bodies()
        
        # update current, target poses to match the new one
        self.current_pose[:3] = pos
        self.current_pose[3:6] = euler_orn
        self.target_pose[:3] = pos
        self.target_pose[3:6] = euler_orn
        
        if not invisible:
            self.update_urdf_hand()
    
    # evaluate the new target position based on where the physical target sphere is (targets follow target bodies)
    def update_targets(self):
        for i, (target_id, _) in enumerate(zip(self.target_body_list, self.targets)):
            self.targets[i] = np.array(p.getBasePositionAndOrientation(target_id)[0])
    
    # put the first n target bodies where the targets are (target bodies follow targets as indicators)
    def reset_target_bodies(self):
        for idx, (end_effector, target) in enumerate(zip(self.end_effectors, self.targets)):
            p.resetBasePositionAndOrientation(self.target_body_list[idx], target, [0.0, 0.0, 0.0, 1.0])

    """
    Move the visible URDF hand to match joint angles
    """
    def update_urdf_hand(self):
        # iterate through the dofs, set hand
        # first 6 dofs for wrist
        pos = [x for x in self.current_pose[:3]]
        # euler angles
        orn = [x for x in self.current_pose[3:6]]
        # convert back
        orn = R.from_euler("XYZ", orn).as_quat()

        cur_pos, cur_orn = p.getBasePositionAndOrientation(self.hand_id)
        p.resetBasePositionAndOrientation(self.hand_id, pos, orn)
        # for each finger
        index = 6
        pose_dofs = []
        joints = get_non_tip_joints(self.hand_id)
        gains = np.ones(len(joints)) * 5
        for finger, full_name in zip(["i","m","p","r","t"], ["index", "middle", "pinky", "ring", "thumb"]):
            basis2 = self.basis[full_name]
            basis1 = self.basis[full_name+"0"] if finger == "t" else self.basis[full_name]
            # knuckle*3, 1, 2
            orn = [x for x in self.current_pose[index:index+3]]
            # convert back
            orn = R.from_euler("XYZ", orn).as_matrix()
            # convert using basis
            orn = basis1 @ orn @ basis1.T
            orn = R.from_matrix(orn).as_quat()
            pose_dofs.append(orn)
            orn = self.current_pose[index+3]
            orn = R.from_euler("x", orn).as_matrix()
            orn = basis2 @ orn @ basis2.T
            orn = R.from_matrix(orn).as_quat()
            pose_dofs.append(orn)
            orn = self.current_pose[index+4]
            orn = R.from_euler("x", orn).as_matrix()
            orn = basis2 @ orn @ basis2.T
            orn = R.from_matrix(orn).as_quat()
            pose_dofs.append(orn)
            index += 5
        p.setJointMotorControlMultiDofArray(self.hand_id, joints, controlMode=p.POSITION_CONTROL, targetPositions=pose_dofs, positionGains=gains)
        #p.resetJointStatesMultiDof(self.hand_id, joints, pose_dofs)
    
    def set_pose(self, pose):
        # convert pose from quat to mat
        pose = R.from_quat(pose).as_matrix()
        # project every mat into its basis
        pose = [b.T @ x @ b for (x,b) in zip(pose, self.basis_list)]
        out = []
        # knuckles have all 3 dofs, remaining 2 have only 1 dof
        for idx in range(len(pose)):
            ex,ey,ez = R.from_matrix(pose[idx]).as_euler("XYZ")
            if idx % 3:
                # only take x component
                out += [ex]
            else:
                # knuckle
                out += [ex,ey,ez]
        # out is now the projected pose
        # prepend the root position
        out = list(self.current_pose[:6]) + out
        # set the joints
        self.current_pose = np.array(out)
        self.target_pose = np.array(out)
        
        self.set_joint_positions()
        self.update_urdf_hand()
        
        # update the targets to match fingertips
        for i, tr in enumerate(self.end_effectors):
            T = tr.global_transform()
            self.targets[i] = T[:3, 3]
        # update the target bodies
        self.reset_target_bodies()
        return out
    def get_joint_positions(self):
        return np.array([jnt_i.get_dof() for jnt_i in self.dof_list])
    
    def ndofs(self):
        return len(self.dof_list)
    
    def compute_jacobian(self, end_effector):
        """ Compute the Jacobian of the end effector.

        Args:
        end_effector: component for which we want to find the Jacobian

        Returns:
        J: Jacobian matrix of shape (3, ndofs)
        """
        global originColVector
        ndofs = len(self.dof_list)
        J = np.zeros((3, ndofs))
        lastNextGuy = end_effector
        lastNextTransform = originColVector
        
        # make sure when traversing backwards we are guaranteed to always encounter DOFs from end to root,
        # otherwise, the DP doesn't work.
        
        # DP for FK too
        fkChains = np.zeros((len(self.transform_list), 4, 4))
        for num in range(len(fkChains)):
            if self.transform_list[num] in end_effector.dependent_transforms:
              fkChains[num] = fkChains[self.transform_list[num].parent._index] @ self.transform_list[num].local_transform() if self.transform_list[num].parent is not None else self.transform_list[num].local_transform()
        for num in range(ndofs-1, -1, -1):
            if self.dof_list[num] in end_effector.dependent_dofs:
                parentChain = fkChains[self.dof_list[num].parent._index] if self.dof_list[num].parent is not None else np.identity(4)#self.dof_list[num].parent.global_transform() if self.dof_list[num].parent is not None else np.identity(4)
                dofDeriv = self.dof_list[num].local_derivative()
                # instead of calculating the chain transform from here to the end, just assume that the last
                # node we visited is this one's direct child, and reuse the previous toEnd, times just the local transform
                # from here to child. Note this fails unless we have strict guarantees about the ordering of the nodes
                # in the dof_list
                toEnd = transform_between(self.dof_list[num], lastNextGuy) @ lastNextTransform
                # IK chain = FK chain from root to here * (my derivative * IK chain from here to end) <- chain rule
                chain = parentChain @ dofDeriv @ toEnd
                # put my IK chain into Jacobian
                J[:, num] = chain[:3]
                # DP
                lastNextGuy = self.dof_list[num]
                lastNextTransform = toEnd
        return J
    def objective_function(self, x):
        """ 
        Objective function of the form 0.5 * L2_norm(error)^2, where the
        error is the difference between the position and the target.

        Args:
        x: Pose

        Returns:
        ret: The value of the objective function.
        """
        self.force_set_joint_positions(x)
        eEStack = np.vstack([eE.global_position() for eE in self.end_effectors])
        targetStack = np.vstack(self.targets)
        # weighted loss
        ret = 0.5 * np.linalg.norm((eEStack - targetStack) * self.weights)**2
        return ret

    def gradient_function(self, x):
        """ 
        Gradient of the objective function above.

        Args:
        x: Pose

        Returns:
        g: The len(x)-sized gradient vector.
        """
        self.force_set_joint_positions(x)
        eEStack = np.vstack([eE.global_position() for eE in self.end_effectors])
        targetStack = np.vstack(self.targets)
        jacobianStack = np.vstack([self.compute_jacobian(eE) for eE in self.end_effectors])
        g = ((eEStack - targetStack) * self.weights).flatten() @ jacobianStack
        return g
    
    def track_targets(self):
        # Solve the pose
        self.reset_target_bodies()
        res = minimize(self.objective_function, 
                       x0=self.target_pose, 
                       jac=self.gradient_function,
                       method='SLSQP', 
                       options={'ftol': 1e-5},
                       bounds=self.bounds)
        
        self.target_pose = res['x']
        
        self.set_joint_positions()
        self.update_urdf_hand()
        """
        for rb in self.body_list:
            rb.update_pybullet()
        """
        return self.target_pose
        
    def get_end_effector_positions(self):
        return np.array([eE.global_position() for eE in self.end_effectors])
    
    def get_joint_position(self, name):
        return self.name_to_transform[name].global_position()
    
    def freeze_target_bodies(self):
        for target_body_id in self.target_body_list:
                p.resetBaseVelocity(target_body_id,[0,0,0],[0,0,0])
                
    def update_self(self):
        self.set_joint_positions()
        self.update_urdf_hand()
        """
        for rb in self.body_list:
            rb.update_pybullet()
        """