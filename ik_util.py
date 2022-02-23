import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
import pickle
import pybullet as p
import glob
import os
import time
from math import sin, cos
from scipy.optimize import minimize
import functools

def make_color(obj_id, color):
  for _id in range(-1, p.getNumJoints(obj_id)):
    p.changeVisualShape(obj_id, _id, rgbaColor=color)
    
# naive palm contact / penetration
def is_contacting(hand_id, obj_id):
  contacts = p.getClosestPoints(hand_id, obj_id, 0.1, 0)
  for c_ in contacts:
    if c_[8] < 0:
      return True
  return False

def get_non_tip_joints(hand_id):
  out = []
  # all link ids starting from 1 (still not counting palm)
  for _id in range(1, p.getNumJoints(hand_id)):
    _name = p.getJointInfo(hand_id, _id)[12].decode('UTF-8')
    if "tip" not in _name:
      out.append(_id)
  return out
  
def disable_collisions(obj_id):
  p.setCollisionFilterGroupMask(obj_id, -1, 0, 0)
  for link in range(p.getNumJoints(obj_id)):
    p.setCollisionFilterGroupMask(obj_id, link, 0, 0)
    
def slerp_single(q1, q2, t):
  q = np.vstack((q1,q2))
  Q = R.from_quat(q)
  keytimes = [0,1]
  output = [t]
  s = Slerp(keytimes, Q)
  interp_frames = s(output)
  return interp_frames.as_quat()[0]
  
def slerp_single_stack(q1, q2, t):
  return [slerp_single(qq1, qq2, t) for qq1, qq2 in zip(q1, q2)]

def lerp(p1, p2, steps=50):
  weights = np.linspace(0,1,steps)
  out = []
  for w in weights:
    out.append(p1 + (p2 - p1) * w)
  return out
  
def lerp_single(p1, p2, t):
  return p1 + (p2 - p1) * t
  
def make_invisible(obj_id):
  make_color(obj_id, (0,0,0,0))
  
def rand_quat():
  r = np.random.randn(4)
  r /= np.linalg.norm(r)
  return r
  
  
def transform_between(start, end):
  if start == end:
    return np.identity(4)
  else:
#     https://stackoverflow.com/a/30629255
    return transform_between(start, end.parent) @ end.local_transform()

def verbose_transform_between(start, end):
  if start == end:
    return np.identity(4)
  else:
#     https://stackoverflow.com/a/30629255
    out = verbose_transform_between(start, end.parent) @ end.local_transform()
    print("trace:", end.local_transform(), "so far:", out)
    return out

class Transformation(object):
  """ Base class for all transformations that we subsequently define. """
  def __init__(self, transform_list, name=None):
    transform_list.append(self)
    self.name = name
    self.parent = None
    self.children = list()
    self.dependent_dofs = set()
    self.dependent_transforms = set()
    self._index = None
  
  def add(self, child):
    child.parent = self
    self.children.append(child)
    return child
  
  def global_transform(self):
    return transform_between(None, self)

  def verbose_global_transform(self):
    return verbose_transform_between(None, self)
  
  def global_position(self):
    return self.global_transform()[:3, 3]

  def local_transform(self):
    assert False and "not implemented" 
  
  def local_derivative(self):
    assert False and "not implemented"
  
  def num_dofs(self):
    assert False and "not implemented"

  def get_dof(self):
    assert False and "not implemented"
  
  def set_dof(self, value=0.0):
    assert False and "not implemented"
  
  def __repr__(self):
    return "%s(%s)" % (self.__class__.__name__, self.name)
 

class Fixed(Transformation):
  """ Fixed transformation. """
  def __init__(self, transform_list, name, tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0):
    super().__init__(transform_list, name=name)
    rot_value = np.identity(4)
    transl_value = np.identity(4)
    rot_value[:3, :3] = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
    transl_value[:3, 3] = (tx, ty, tz)
    self._value = transl_value @ rot_value

  def num_dofs(self):
    return 0

  def local_transform(self):
    return self._value


class Translation(Transformation):
  """ Class for translation. """
  def __init__(self, transform_list, name, axis, value=0.0):
    super().__init__(transform_list, name=name)
    self._value = value
    self.axis = axis
    self.axis_index = "xyz".find(axis)
    self.T = np.identity(4)
    self.dT = np.zeros((4, 4))
  
  def num_dofs(self):
    return 1
  
  def get_dof(self):
    return self._value

  def set_dof(self, value=0.0):
    self._value = value
  
  def local_transform(self):
    self.T[self.axis_index, 3] = self._value
    return self.T

  def local_derivative(self):
    self.dT[self.axis_index, 3] = 1.0
    return self.dT
    
class Hinge(Transformation):
  """Class for rotation. """
  def __init__(self, transform_list, name, axis, theta=0.0):
    super().__init__(transform_list, name=name)
    self._theta = theta
    self.axis = axis
    self.axis_index = "xyz".find(axis)    
  
  def num_dofs(self):
    return 1

  def get_dof(self):
    return self._theta
  
  def set_dof(self, value=0.0):
    self._theta = value
  
  def local_transform(self):
    """
    Returns:
      transform: The 4x4 transformation matrix.
    """
    cth, sth = cos(self._theta), sin(self._theta)
    out = np.identity(4)
    c = np.cos(self._theta)
    s = np.sin(self._theta)
    if self.axis == 'x':
        out[:3,:3] = np.array([[1, 0, 0],[0, c, -s],[0, s, c]])
    elif self.axis == 'y':
        out[:3,:3] = np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]])
    else:
        out[:3,:3] = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])
    return out
    
  def local_derivative(self):
    """
    Returns:   
      deriv_transform: The 4x4 derivative matrix.
    """
    
    out = np.zeros((4,4))
    c = np.cos(self._theta)
    s = np.sin(self._theta)
    if self.axis == 'x':
        out[:3,:3] = np.array([[0, 0, 0],[0, -s, -c],[0, c, -s]])
    elif self.axis == 'y':
        out[:3,:3] = np.array([[-s, 0, c],[0, 0, 0],[-c, 0, -s]])
    else:
        out[:3,:3] = np.array([[-s, -c, 0],[c, -s, 0],[0, 0, 0]])
    return out
    

def smooth_damp(current, target, currentVelocity, smoothTime, maxSpeed, dT):
    smoothTime = max(0.0001, smoothTime)
    omega = 2 / smoothTime
    x = omega * dT
    exp = 1 / (1 + x + 0.48 * x**2 + 0.235 * x**3)
    change = current - target
    originalTo = target
    
    maxChange = maxSpeed * smoothTime
    change = np.clip(change, -maxChange, maxChange)
    target = current - change
    
    temp = (currentVelocity + omega * change) * dT
    currentVelocity = (currentVelocity - omega * temp) * exp
    output = target + (change + temp) * exp
    
    condition = np.logical_not(np.logical_xor(originalTo - current > 0, output > originalTo))
    candidateVelocity = (output - originalTo) / dT
    
    output = np.where(condition, originalTo, output)
    currentVelocity = np.where(condition, candidateVelocity, currentVelocity)
    
    return output, currentVelocity
    
    
body_list = list()
target_body_list = list()

class RigidBody(object):
  """
  A base class for rigid bodies.
  """
  def __init__(self, 
      body_list, transform):
    self.transform = transform
    self.uid = -1
    body_list.append(self)

  
  def get_position_and_orientation(self):
    T = self.transform.global_transform()
    position = T[:3, 3]
    orientation = R.from_matrix(T[:3, :3]).as_quat()
    return position, orientation

  def update_pybullet(self):
    position, orientation = self.get_position_and_orientation()
    p.resetBasePositionAndOrientation(self.uid, position, orientation)


class Box(RigidBody):
  """ We use this class, which inherits from the RigidBody class, to create the 
  torso and limbs of our character.
  """
  def __init__(self,
      body_list, transform,
      color = np.array([0, 1, 1, 1]),
      half_extends = [1, 1, 1],
      texUid = None):
    super().__init__(body_list,transform)

    self.visualId = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extends, rgbaColor=color, specularColor=[1, 1, 1])
    position, orientation = self.get_position_and_orientation()
    self.uid = p.createMultiBody(1.0, -1, self.visualId, position, orientation)
    if texUid:
        p.changeVisualShape(self.uid, -1, textureUniqueId=texUid)

class Sphere(RigidBody):
  """ We use this class, which inherits from the RigidBody class, to create the 
  hands and face of our character.
  """
  def __init__(self,
      body_list, transform,
      color = np.array([0, 1, 1, 1]),
      radius = 1.0):
    super().__init__(body_list,transform)
    self.radius = radius

    self.visualId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, specularColor=[1, 1, 1])
    position, orientation = self.get_position_and_orientation()
    self.uid = p.createMultiBody(1.0, -1, self.visualId, position, orientation)

def reset_simulation():
    time = 0
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.setTimeStep(1/240) # Dummy time step
    body_list = list()   