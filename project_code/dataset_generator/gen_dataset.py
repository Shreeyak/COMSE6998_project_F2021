from camera import Camera, save_obs
import objects
import pybullet_data
import pybullet as p
import sys
sys.path.insert(1, './')
from rotation_generator import RotationGenerator
import os
from typing import List, Dict


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

    def generate_dataset(self)->Dict[int,Tuple(np.array,Dict[str,List[str]])]:
        """
        Generates the dataset of our project.

        Returns:
            training_pairs: An object which stores the training pairs of our
                            training data. An observation constituted by an rgb
                            image and a depth mask image is made before and
                            after each random transformation is applied; the
                            names of those observation filenames are recorded.
                            The return object stores them in a dictionary
                            whose keys are the scene number and whose values
                            are tuples of the transformation matrix and
                            associated before & after image pairs. One key-val
                            pair, corresponding to the first scene, might look
                            like
                            {i:
                                (np.array([1, 0, 0], [0, 1, 0], [0, 0, 1]),
                                {'rgb': ["1_before_rgb.png", "1_after_rgb.png"],
                                 'mask': ["1_before_gt.png", "1_after_gt.png"]}
                                )
                            }
        """
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
        training_pairs = {}
        for i in range(0, self.training_scenes):
            print(f'==> {i} / {self.training_scenes}')
            # Reset object(s) by dropping on the ground
            objects.reset_obj(
                obj_ids,
                self.obj_positions,
                self.obj_orientations,
                scene_id = i
            )
            # save an observation pre-transformation matrix
            rgb1,mask1 = save_obs(self.dataset_dir,self.this_camera,i,"before")
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
            rgb2,mask2 = save_obs(self.dataset_dir,self.this_camera,i,"after")
            # Save transformation matrix as label for this scene
            observations = {'rgb': [rgb1, rgb2],
                            'mask': [mask1, mask2]}
            training_pairs[i] = (rot_mat, observations)

        p.disconnect()
        return training_pairs


def main():
    data_gen = DatasetGenerator(
        training_scenes=30,
        obj_foldernames=["011_banana"],
        obj_positions=[[0.0, 0.0, 0.0]],
        dataset_dir="../dataset/train"
    )
    training_pairs = data_gen.generate_dataset()


if __name__ == '__main__':
    main()
