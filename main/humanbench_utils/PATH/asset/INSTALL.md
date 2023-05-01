# Installation

### Requirements
* Linux, CUDA>=9.2, GCC>=5.4
* PyTorch >= 1.8.1
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```
    
### Setup for COCO Pose Estimation Evalutaion

Install [mmpose](https://github.com/open-mmlab/mmpose) following the instructions in [here](https://mmpose.readthedocs.io/en/v0.29.0/install.html). 
Or simply use the following command.
<!-- * Note we use mmpose @ `8c58a18b` -->
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```
