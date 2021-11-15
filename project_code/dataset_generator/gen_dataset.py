import os

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

    def __init__(self):
        """
        Initializes the DatasetGenerator class.

        Args: none
        """
        pass

    @staticmethod
    def generate_dataset():
        physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Set up camera
        this_camera = Camera(
            image_size=(240, 320),
            near=0.01,
            far=10.0,
            fov_w=69.40
        )
        # Define number of training scenes
        training_scene = 30  # TODO: replace with actual value

        # Number of observations to be made in each scene (we only
        # need the beginning and the end
        num_observations = 2

        # Load floor
        plane_id = p.loadURDF("plane.urdf")

        # Load object(s)
        # Future direction: include more objects here
        list_obj_foldername = [
            "011_banana",
        ]
        num_obj = len(list_obj_foldername)
        list_obj_position = [[0.1, 0.1, 0.1]]
        list_obj_orientation = objects.gen_obj_orientation(
            num_scene=training_scene,
            num_obj=num_obj
        )
        list_obj_id = obj.load_obj(
            list_obj_foldername,
            list_obj_position,
            list_obj_orientation,
        )

        # Generate training data set
        dataset_dir = "../dataset/train"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            os.makedirs(dataset_dir + "rgb/")
            os.makedirs(dataset_dir + "gt/")
        print("Start generating the training set.")
        print(f'==> 1 / {training_scene}')
        save_obs(
            dataset_dir,
            this_camera,
            num_obs=num_observations,
            scene_id=0
        )
        for i in range(1, training_scene):
            print(f'==> {i+1} / {training_scene}')
            objects.reset_obj(
                list_obj_id,
                list_obj_position,
                list_obj_orientation,
                scene_id=i
            )
            save_obs(
                dataset_dir,
                this_camera,
                num_obs=num_observations,
                scene_id=i
            )

        p.disconnect()
