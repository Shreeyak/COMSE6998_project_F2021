from camera import Camera, save_obs
import objects
import pybullet_data
import pybullet as p
import rotation_generator
import os
from typing import List
import sys
sys.path.insert(1, '../')


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

        # Load floor
        plane_id = p.loadURDF("plane.urdf")
        obj_ids = objects.load_obj(self.obj_foldernames,self.obj_positions,self.obj_orientations)
        

        print("Start generating the training set.")
        print(f'==> 1 / {self.training_scenes}')
        save_obs(self.dataset_dir, self.this_camera, scene_id=0)

        """
        TODO: control flow should be roughly

        reset the object, save_observation, get the current position &
        orientation, get a randomly generated transformation matrix, apply it
        to current position, then save_observation again
        """
        for i in range(1, self.training_scenes):
            print(f'==> {i+1} / {self.training_scenes}')
            objects.reset_obj(
                self.obj_ids,
                self.obj_positions,
                self.obj_orientations,
                scene_id=i
            )
            save_obs(self.dataset_dir, self.this_camera, scene_id=i)

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
