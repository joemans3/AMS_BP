# Documentation for the simulation configuration file of the same name 
* [Simulation Configuration File](sim_config.json)
* Latest version supported: v.0.1

## Simulation Configuration File
* version: string
    * version of the simulation configuration file
* length_unit: string
    * length unit of the simulation (e.g. nm, um, mm)
* space_unit: string
    * space unit of the simulation (this is just pixel, should not change)
* time_unit: string
    * time unit of the simulation (e.g. s, ms, us)
* intensity_unit: string
    * intensity unit of the simulation (AUD only supported)
* diffusion_unit: string
    * diffusion unit of the simulation (e.g. um^2/s, mm^2/s)
    
* Cell_Parameters: dict  
    * cell_space: 2D array (units of space_unit)
        1. cell_space[0]: x coordinates of the cell space (min, max)
        2. cell_space[1]: y coordinates of the cell space (min, max)
    * cell_axial_radius: float (units of space_unit)
        1. The distance from z=0 in either direction that the cell extends 
    * number_of_cells: int
        1. number of cells to simulate (if more than 1 hen all are simulated in one folder defined by the output_path)

* Track_Parameters: dict  
    * num_tracks: int
        1. number of tracks to simulate
    * track_type: string
        1. type of track to simulate ("fbm")
    * track_length_mean: int (frames) 
        1. mean length of the track
    * track_distribution: string
        1. distribution of the track lengths ("exponential","constant")
    * diffusion_coefficient: list of floats (units of diffusion_unit)
        1. diffusion coefficient of the track, the length of the list is the unique type of diffusion coefficients
    * diffusion_track_amount: list of floats
        1. Only viable if allow_transition_probability is False
        2. length is the total number of diffusion coefficients
        3. each element is the probability of the track having the diffusion coefficient at the same index in the diffusion_coefficient list (add up to 1.0)
    * hurst_exponent: list of floats
        1. hurst exponent of the track, the length of the list is the unique type of hurst exponents
    * hurst_track_amount: list of floats
        1. Only viable if allow_transition_probability is False
        2. length is the total number of hurst exponents
        3. each element is the probability of the track having the hurst exponent at the same index in the hurst_exponent list (add up to 1.0)
    * allow_transition_probability: bool
        1. whether to allow transition probabilities between different diffusion coefficients and hurst exponents within a track
        2. if false, the track will have a single diffusion coefficient and hurst exponent
    * transition_matrix_time_step: int
        1. time step at which the diffusion and hurst exponent transition matrices are supplied in the following parameters
        2. the units are in time_unit (so 100 ms would be 100)
    * diffusion_transition_matrix: 2D array (discrete state probabilitiy at the transition_matrix_time_step = dt)
        1. transition matrix between different diffusion coefficients
        2. rows are the current diffusion coefficient
        3. columns are the next diffusion coefficient
        4. rows must sum to 1.0
    * hurst_transition_matrix: 2D array (discrete state probabilitiy at the transition_matrix_time_step = dt)
        1. transition matrix between different hurst exponents
        2. rows are the current hurst exponent
        3. columns are the next hurst exponent
        4. rows must sum to 1.0
    * state_probability_diffusion: 1D array (probability)
        1. probability of a track being in a certain diffusion coefficient state
        2. length is the number of unique diffusion coefficients
    * state_probability_hurst: 1D array (probability)
        1. probability of a track being in a certain hurst exponent state
        2. length is the number of unique hurst exponents

* Global_Parameters: dict  
    * field_of_view_dim: 1D array (units of space_unit)
        1. field of view dimensions (x,y (pixels))
    * frame_count: int
        1. number of frames to simulate
    * exposure_time: float or int (units of time_unit)
        1. exposure time of the camera
    * interval_time: float or int (units of time_unit)
        1. time between frames that the camera is on
    * oversample_motion_time: float or int (units of time_unit)
        1. oversampling the motion for motion blur
        2. if oversample_motion_time == frame_time == exposure_time, then there is no motion blur
        3. cannot be greater than frame_time or exposure_time
    * pixel_size: float (units of length_unit)
        1. size of the pixel
    * axial_detection_range: float (units of length_unit)
        1. from z=0, the distance in either direction that the camera can detect a single molecule excitation
    * base_noise: float (units of intensity_unit)
        1. base noise of the camera (offset)
    * point_intensity: float (units of intensity_unit)
        1. intensity of a single molecule excitation
    * psf_sigma: float (units of length_unit)
        1. size of the psf (assumed to be gaussian)
    * axial_function: string ("exponential","ones"(no effect))
        1. function used to determine how the intensity of the single molecule changes with z

* Condensate_Parameters: dict  
    * initial_centers: 2D array (units of space_unit)
        1. initial centers of the condensates
        2. [x,y,z] coordinates per row
        3. number of rows is the number of condensates
    * initial_scale: 1D (units of space_unit)
        1. initial radius of the condensates
        2. number of elements is the number of condensates
        3. must be the same length as initial_centers
    * diffusion_coefficient: 1D array (units of diffusion_unit)
        1. diffusion coefficient of the condensates
        2. number of elements is the number of condensates
        3. must be the same length as initial_centers
    * hurst_exponent: 1D array
        1. hurst exponent of the condensates
        2. number of elements is the number of condensates
        3. must be the same length as initial_centers
    * density_dif: float
        1. density difference between the condensates and the rest of the cell

* Output_Parameters: dict  
    * output_path: string
        1. path to save the output, directory
    * output_name: string
        1. name of the output file
    *subsegment_type: string
        1. function used to do projections ("mean","max","sum")
    * subsegment_number: int
        1. number of subsegments to divide the cell frames into
        2. if total movie is 500 frames and this is 5 then there will be 100 frames per subsegment and 5 subsegments in total
        3. Make sure that the total number of frames is divisible by the number of subsegments (modulus is 0)
