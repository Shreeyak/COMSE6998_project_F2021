"""
rotation_generator.py: A library to generate small 3D rotation matrices

The class generates a 3x3 matrix describing a general rotation in 3D
space, where each element relevant to the rotation is defined relative
to angles (in radians) of rotation for roll, pitch, and yaw, all of
which have 0 (inclusive) minima and user-specified maxima (inclusive).

This is intended to create "small" rotations, so we recommend
limiting the input angle maximum to π/4 (~0.785398) radians.
"""

import numpy as np
import random

class RotationGenerator:

    def __init__(self, angle_max: float):
        """
        init function for the RotationGenerator class.

        Args:
            angle_max: The maximum radian value for all randomly generated
                       angles of rotation.
        """
        self.angle_max = angle_max

    def generate_simple_rotation() -> np.array:
        """
        A method to generate a random simple 3D rotation matrix.

        A simple 3D rotation is a rotation by one angle theta about one axis, which here is also randomly chosen.

        Return: A 3x3 matrix describing a simple rotation transformation.
        """
        theta = random.uniform(0, self.angle_max)
        cos_val, sin_val = np.cos(theta), np.sin(theta)

        # Rotation about x
        r_x = np.array([[1,       0,        0],
                        [0, cos_val, -sin_val],
                        [0, sin_val, cos_val]])

        # Rotation about y
        r_y = np.array([[cos_val,  0, sin_val],
                        [0      ,  1,       0],
                        [-sin_val, 0, cos_val]])

        # Rotation about z
        r_z = np.array([[cos_val, -sin_val, 0],
                        [sin_val,  cos_val, 0],
                        [0      ,  0      , 1]])

        return random.choice([r_x, r_y, r_z])


    def generate_rotation() -> np.array:
        """
        A method to generate a random general 3D rotation matrix.

        A general 3D rotation matrix is a rotation consisting of roll, pitch,
        and yaw angles about all three axes. It can be broken down into a matrix
        product of rotations about x by roll angle gamma, about y by pitch angle
        beta, and about z by yaw angle alpha.

        Return: A 3x3 matrix describing a general rotation transformation.
        """
        rot_angles = np.random.uniform(low=0.0, high=self.angle_max, size=3)
        alpha, beta, gamma = rot_angles[0], rot_angles[1], rot_angles[2]

        return np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
        [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
        [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
