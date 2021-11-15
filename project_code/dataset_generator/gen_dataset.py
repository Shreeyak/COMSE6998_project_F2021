import os
from typing import List

import pybullet as p
import pybullet_data

import objects
from camera import Camera, save_obs


class DatasetGenerator():
    """
    A class which generates the dataset for our project. Connects to
    pybullet and ____

    # TODO: fill out this docstring a lot better
    """

    def __init__(self,
        training_scenes: int,
        num_observations: int,
        obj_foldernames: List[str],
        obj_positions: List[List[float]],
        dataset_dir: str):
        """
        Initializes the DatasetGenerator class.

        Args:
            training_scenes: The number of scenes we'd like to give our model
                             to train on. Each consists of an object or objects
                             being rotated according to a random rotation
                             matrix.
            num_observations: The number of observations per scene (i.e.
                              observations of the rotation being applied).
        """
        self.this_camera = Camera(
            image_size = (240, 320),
            near = 0.01,
            far = 10.0,
            fov_w = 69.40
        )
        self.training_scenes = training_scenes
        self.num_observations = num_observations
        self.obj_foldernames = [fn for fn in obj_foldernames]
        self.obj_positions = obj_positions
        self.obj_orientations = objects.gen_obj_orientation(
            num_scene = self.training_scenes,
            num_obj = len(self.obj_foldernames)
        )
        self.obj_ids = objects.load_obj(
            self.obj_foldernames,
            self.obj_positions,
            self.obj_orientations
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

        print("Start generating the training set.")
        print(f'==> 1 / {self.training_scenes}')
        save_obs(
            self.dataset_dir,
            self.this_camera,
            num_obs=self.num_observations,
            scene_id=0
        )
        for i in range(1, self.training_scenes):
            print(f'==> {i+1} / {self.training_scenes}')
            objects.reset_obj(
                self.obj_ids,
                self.obj_positions,
                self.obj_orientations,
                scene_id=i
            )
            save_obs(
                self.dataset_dir,
                self.this_camera,
                num_obs=self.num_observations,
                scene_id=i
            )
        p.disconnect()


def main():
    data_gen = DatasetGenerator(
        training_scenes = 30,
        num_observations = 2,
        obj_foldernames = ["011_banana"],
        obj_positions = [[0.1, 0.1, 0.1]],
        dataset_dir = "../dataset/train"
    )
    data_gen.generate_dataset()


if __name__ == '__main__':
    main()
