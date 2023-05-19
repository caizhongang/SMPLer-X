# Installation

Before installing anything please make sure to set the environment variable
*$CUDA_SAMPLES_INC* to the path that contains the header `helper_math.h`, which
can be found in the repo [CUDA Samples repository](https://github.com/NVIDIA/cuda-samples).
To install the module run the following commands:  

**1. Clone this repository**
```Shell
git clone https://github.com/vchoutas/torch-mesh-isect
cd torch-mesh-isect
```
**2. Install the dependencies**
```Shell
pip install -r requirements.txt 
```
**3. Run the *setup.py* script**
```Shell
python setup.py install
```

## Dependencies

1. [PyTorch](https://pytorch.org)

### Optional Dependencies

1. [Trimesh](https://trimsh.org) for loading triangular meshes
2. [open3d](http://www.open3d.org/) for visualization

The code has been tested with Python 3.6, CUDA 10.0, CuDNN 7.3 and PyTorch 1.0.
