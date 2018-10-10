# Exploring Spatial Context for 3D Semantic Segmentation of Point Clouds
Created by Francis Engelmann, Theodora Kontogianni, Alexander Hermans, Jonas Schult and Bastian Leibe 
from RWTH Aachen University.

![prediction example](doc/exploring_header.png?raw=True "dfdf")

### Introduction
This work is based on our paper 
[Exploring Spatial Context for 3D Semantic Segmentation of Point Clouds](https://www.vision.rwth-aachen.de/media/papers/PID4967025.pdf),
which appeared at the IEEE International Conference on Computer Vision (ICCV) 2017, 3DRMS Workshop. 

You can also check our [project page](https://www.vision.rwth-aachen.de/page/3dsemseg) for further details.

Deep learning approaches have made tremendous progress in the field of semantic segmentation over the past few years. However, most current approaches operate in the 2D image space. Direct semantic segmentation of unstructured 3D point clouds is still an open research problem. The recently proposed PointNet architecture presents an interesting step ahead in that it can operate on unstructured point clouds, achieving decent segmentation results. However, it subdivides the input points into a grid of blocks and processes each such block individually. In this paper, we investigate the question how such an architecture can be extended to incorporate larger-scale spatial context. We build upon PointNet and propose two extensions that enlarge the receptive field over the 3D scene. We evaluate the proposed strategies on challenging indoor and outdoor datasets and show improved results in both scenarios.

In this repository, we release code for training and testing various pointcloud semantic segmentation networks on
arbitrary datasets.

### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{3dsemseg_ICCVW17,
      author    = {Francis Engelmann and
                   Theodora Kontogianni and
                   Alexander Hermans and
                   Bastian Leibe},
      title     = {Exploring Spatial Context for 3D Semantic Segmentation of Point Clouds},
      booktitle = {{IEEE} International Conference on Computer Vision, 3DRMS Workshop, {ICCV}},
      year      = {2017}
    }

   
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>.
The code has been tested with Python 3.6 and TensorFlow 1.8.

### Usage
In order to get more representative blocks, it is encouraged to uniformly downsample the original point clouds.
This is done via the following script:

    python tools/downsample.py --data_dir path/to/dataset --cell_size 0.03

This statement will produce pointclouds where each point will be representative for its 3cm x 3cm x 3cm neighborhood.

To train/test a model for semantic segmentation on pointclouds, you need to run:

    python run.py --config path/to/config/file.yaml
    
Detailed instruction of the structure for the yaml config file can be found in the wiki.
Additionally, some example configuration files are given in the folder `experiments`.

Note that the final evaluation is done on the full sized point clouds using k-nn interpolation.

### Reproducing the scores of our paper for stanford indoor 3d

#### Downloading the data set
First of all, Stanford Large-Scale 3D Indoor Spaces Dataset has to be downloaded.
Follow the instructions [here](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1).
The aligned version 1.2 is used for our results.

#### Producing numpy files from the original dataset
Our pipeline cannot handle the original file type of s3dis. So, we need to convert it to npy files.
Note that Area_5/hallway_6 has to be fixed manually due to format inconsistencies.

    python prepare_s3dis.py --input_dir path/to/dataset --output_dir path/to/output

#### Downsampling for training
Before training, we downsampled the pointclouds.

    python tools/downsample.py --data_dir path/to/dataset --cell_size 0.03

#### Training configuration scripts
Configuration files for all experiments are located in `experiments/iccvw_paper_2017/*`. For example, they can be
launched as follows:

    python run.py --config experiments/iccvw_paper_2017/s3dis_mscu/s3dis_mscu_area_1.yaml
    
The above script will run our multi scale consolidation unit network on stanford indoor 3d with test area 1.

#### Evaluating on full scale point clouds
Reported scores on the dataset are based on the full scale pointclouds.
In order to do so, we need to load the trained model and set the `TEST` flag.

Replace `modus: TRAIN_VAL` with 

```yaml
    modus: TEST
    model_path: 'path/to/trained/model/model_ckpts'
```
which is located in the log directory specified for training.


### VKitti instructions
* coming soon ...

### Trained models for downloading
* Coming soon ...

### License
Our code is released under MIT License (see LICENSE file for details).
