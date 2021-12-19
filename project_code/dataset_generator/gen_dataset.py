from camera import Camera, save_obs
import objects
import pybullet_data
import pybullet as p
import sys
sys.path.insert(1, './')
from rotation_generator import RotationGenerator
import os
from typing import List, Dict, Tuple
import numpy as np
import re
from scipy.spatial.transform import Rotation as R


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
            image_size=(64, 64),
            near=0.01,
            far=10.0,
            fov_w=69.40
        )
        """ Training scenes from init, how many pictures to take"""
        self.training_scenes = training_scenes
        self.obj_foldernames = [fn for fn in obj_foldernames]
        self.obj_positions = obj_positions
        self.obj_orientations = objects.gen_obj_orientation(
            num_scene=self.training_scenes,
            num_obj=1
        )

        self.dataset_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            os.makedirs(dataset_dir + "/rgb/")
            os.makedirs(dataset_dir + "/gt/")
            os.makedirs(dataset_dir + "/depth/")
            os.makedirs(dataset_dir + "/rotations/")
            
        self.rot_file = dataset_dir+"/rotations/rotations.csv"
       
        f = open(self.rot_file, 'w') 
       
        #if os.path.exists(self.rot_file):
           # os.remove(self.rot_file)
            
        

    def generate_dataset(self)->Dict[int,Tuple[np.array,Dict[str,List[str]]]]:
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
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #dont need gravity, can spawn half a meter
        p.setGravity(0, 0, 0)
        rot_gen = RotationGenerator(0.7853)

        # Load floor
        p.loadURDF("plane.urdf")
        
        f = open(self.rot_file, 'a')
            
        current_file_num = 0
        training_pairs = {}
        print("Start generating the training set.")
         
        for j in range(0,len(self.obj_foldernames)):
            
            #for each object folder in the list, generate traning scenes
            current_obj = [self.obj_foldernames[j]]
            
            #get new orientations
            self.obj_orientations = objects.gen_obj_orientation(
                num_scene=self.training_scenes,
                num_obj=1
            )
            
            # Load current object
            obj_ids = objects.load_obj(
                current_obj,
                self.obj_positions,
                self.obj_orientations)

            for i in range(0, self.training_scenes):
                print(f'==> {i+1} / {self.training_scenes}')
                # Reset object(s) by dropping on the ground
                objects.reset_obj(
                    obj_ids,
                    self.obj_positions,
                    self.obj_orientations,
                    scene_id =  i
                )
                      
                # save an observation pre-transformation matrix
                rgb1,mask1,depth1 = save_obs(self.dataset_dir,self.this_camera,i+current_file_num,"before")
                
                # collect current position and orientation info
                objPos, objOrn = p.getBasePositionAndOrientation(obj_ids[0])
                
                # generate a random 3D rotation quaternion
                rot_mat, rot_quat = rot_gen.generate_simple_rotation()
                
                # apply rotation matrix to object's position and orientation
                curRotMat = np.array(p.getMatrixFromQuaternion(objOrn)).reshape(3,3)
                newRotMat = curRotMat @ rot_mat
                newQuat = R.from_matrix(newRotMat).as_quat()
                
                # Reset the object's position with respect to new values
                p.resetBasePositionAndOrientation(
                    obj_ids[0], # currently only works for the banana
                    posObj=objPos,
                    ornObj=newQuat
                )
    
                #Save rot_matrix after making it flat  
                rot_flat = str(rot_mat.flatten())
                rot_clean_str = " "
                rot_clean_str = rot_clean_str.join(rot_flat.split())
                rot_clean_str = rot_clean_str[2:-1]
                f.write(rot_clean_str)
                f.write('\n')
                
                # save an observation post-transformation matrix
                rgb2,mask2,depth2 = save_obs(self.dataset_dir,self.this_camera,i+current_file_num,"after")
                
            current_file_num+=60
            
            p.removeBody(obj_ids[0])

        p.disconnect()
        f.close()

def main():
    data_gen = DatasetGenerator(
        training_scenes=60,
        obj_foldernames=["004_sugar_box","005_tomato_soup_can","007_tuna_fish_can","011_banana","024_bowl"],
        obj_positions=[[0.0, 0.0, 0.1]],
        dataset_dir="../dataset/train1"
    )
    training_pairs1 = data_gen.generate_dataset()
    
    

if __name__ == '__main__':
    main()
