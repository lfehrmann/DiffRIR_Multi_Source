import numpy as np
import config
import trace as G
import rooms.dataset as dataset


"""
Importing this document automatically loads data from the shoebox dataset
"""

speed_of_sound = 343

"""
Locations of all surfaces in Meters
"""
max_x = 6.0
max_y = 11.75
max_z = 2.8

rear_wall = G.Surface(np.array([[0, 0, 0], 
                                  [0, 0, max_z],
                                  [max_x, 0, 0]]))

front_wall = G.Surface(np.array([[0, max_y, 0],
                                  [max_x, max_y, 0],
                                  [0, max_y, max_z]]))


floor = G.Surface(np.array([[0, 0, 0],
                                [max_x, 0, 0],
                                [0, max_y, 0]]))


ceiling = G.Surface(np.array([[0, 0, max_z],
                                [0, max_y, max_z],
                               [max_x, 0, max_z]]))

left_wall = G.Surface(np.array([[0, 0, 0],
                                [0, max_y, 0],
                                 [0, 0, max_z]]))

right_wall = G.Surface(np.array([[max_x, 0, 0],[max_x, 0, max_z],                               
                                [max_x, max_y, 0]]))


walls = [rear_wall, front_wall, floor, ceiling, left_wall, right_wall]

base_surfaces = walls


"""
Train and Test Split
"""

train_indices = [0, 7, 18, 45, 56, 63, 64, 71, 83, 108, 120, 127]
valid_indices = dataset.compute_complement_indices(list(train_indices), 128)


#Speaker xyz from source_shifted_*
BaseDataset = dataset.Dataset(
   load_dir = config.shoebox15_path,
   speaker_xyz= np.array([3.1215, 8.8979, 1.45]),
   all_surfaces = base_surfaces,
   speed_of_sound = speed_of_sound,
   default_binaural_listener_forward = np.array([0,1,0]),
   default_binaural_listener_left = np.array([-1,0,0]),
   parallel_surface_pairs=[[0,1], [2,3], [4,5]],
   train_indices = train_indices,
   valid_indices = valid_indices,
   max_order = 5,
   max_axial_order = 10,
   n_data = 128
)
