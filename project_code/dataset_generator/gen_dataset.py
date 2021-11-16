from camera import Camera, save_obs
import objects
import pybullet_data
import pybullet as p
import sys
sys.path.insert(1, './')
from rotation_generator import RotationGenerator
import os
from typing import List


class DatasetGenerator(object):
    """
    A class which generates the data set for our project. The data set consists
    of scenes (pairs of images) each labeled by a rotation matrix. In a given
    scene, an object is dropped from an arbitrary height onto to the ground in
    a pybullet physics simulation. At that point, an observation is made (a
    picture is taken). Then, a randomly generated rotation matrix is applied to
    the object and another observation is made. Our model will be attempting to
    regress the rotation matrix that caused the transformation of the object
    from the first observation to the second observations.
    """

    def __init__(self,
                 training_scenes: int,
                 obj_foldernames: List[str],
                 obj_positions: List[List[float]],
                 dataset_dir: str):
        """
        Initializes the DatasetGenerator class.

        Args:
            training_scenes: The number of scenes we'd like to give our model
                             to train on.
            obj_foldernames: The names of each object folder (located in
                             the YCB_subsubset directory of dataset_generator
                             folder).
            obj_positions: A list of the initial positions for each object,
                           each given as 3-vectors of Euclidean x, y, z
                           coordinates.
            dataset_dir: The directory we'd like to save our training examples
                         to.
        """
        self.this_camera = Camera(
            image_size=(240, 320),
            near=0.01,
            far=10.0,
            fov_w=69.40
        )
        self.training_scenes = training_scenes
        self.obj_foldernames = [fn for fn in obj_foldernames]
        self.obj_positions = obj_positions
        self.obj_orientations = objects.gen_obj_orientation(
            num_scene=self.training_scenes,
            num_obj=len(self.obj_foldernames)
        )

        self.dataset_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            os.makedirs(dataset_dir + "rgb/")
            os.makedirs(dataset_dir + "gt/")

    def generate_dataset(self):
        physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        rot_gen = RotationGenerator(0.7853)

        # Load floor
        plane_id = p.loadURDF("plane.urdf")

        # Load objects
        obj_ids = objects.load_obj(
            self.obj_foldernames,
            self.obj_positions,
            self.obj_orientations)


        print("Start generating the training set.")

        """
        TODO: control flow should be roughly

        reset the object, save_observation, get the current position &
        orientation, get a randomly generated transformation matrix, apply it
        to current position, then save_observation again
        """
        for i in range(1, self.training_scenes+1):
            print(f'==> {i} / {self.training_scenes}')
            # Reset object(s) by dropping on the ground
            objects.reset_obj(
                obj_ids,
                self.obj_positions,
                self.obj_orientations,
                scene_id = i
            )

            # save an observation pre-transformation matrix
            save_obs(self.dataset_dir, self.this_camera, i, "before")

            # collect current position and orientation info
            # currently only works with one object (the banana)
            objPos, objOrn = p.getBasePositionAndOrientation(self.obj_ids[0])

            # generate a random 3D rotation matrix
            rot_mat = rot_gen.generate_rotation()

            # apply rotation matrix to object's position and orientation
            newPos = rot_mat@objPos
            curEul = p.getEulerFromQuaternion(objOrn)
            newOrn = p.getQuaternionFromEuler(rot_mat@curEul)

            # Reset the object's position with respect to new values
            # Currently only works for one object (the banana)
            p.resetBasePositionAndOrientation(
                self.obj_ids[0],
                posObj=newPos,
                ornObj=newOrn
            )

            # save an observation post-transformation matrix
            save_obs(self.dataset_dir, self.this_camera, i, "after")

            # TODO: save transformation matrix as label for this scene



        p.disconnect()


def main():
    data_gen = DatasetGenerator(
        training_scenes=30,
        obj_foldernames=["011_banana"],
        obj_positions=[[0.0, 0.0, 0.0]],
        dataset_dir="../dataset/train"
    )
    data_gen.generate_dataset()


"""
TODO IN GEN_DATASET:
    include in the data reset_obj and save_obs loop, we need to get the
    current orientation of the object, save an observation, then apply the
    transformation matrix to the current orientation, save another observation,
    and then go to next scene (save rotation as a label somehow)
"""


if __name__ == '__main__':
    main()
