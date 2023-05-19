# Computing mesh-mesh intersection

This package provides a PyTorch module that can efficiently compute mesh-mesh
intersections using a BVH.


## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
  * [Examples](#examples)
  * [Citation](#citation)
  * [License](#license)
  * [Contact](#contact)

## Description

This repository provides a PyTorch wrapper around a CUDA kernel that implements
the method described in [Maximizing parallelism in the construction of BVHs,
octrees, and k-d trees](https://dl.acm.org/citation.cfm?id=2383801). More
specifically, given an input mesh it builds a
BVH tree for each one and queries it for self-intersections. 

## Installation

See the instructions [here](docs/install.md) on how to install the package.

## Examples

### Fitting to measurements

To fit a 3D human body model to height, weight and circumenference measurements
use the following command:
```python
python examples/fit_measurements.py --model-folder PATH_TO_BODY_MODELS \
    --model-type [smpl/smplh/star/smplx] --gender neutral/female/male --num-betas 30 \
    --meas-vertices-path data/smpl_measurement_vertices.yaml
```
If you are using SMPL-X then set `--meas-vertices-path data/smplx_measurements.yaml`.

## Citation

If you find this code useful in your research please cite the relevant work(s) of the following list, for detecting and penalizing mesh intersections accordingly:

```
@inproceedings{Karras:2012:MPC:2383795.2383801,
    author = {Karras, Tero},
    title = {Maximizing Parallelism in the Construction of BVHs, Octrees, and K-d Trees},
    booktitle = {Proceedings of the Fourth ACM SIGGRAPH / Eurographics Conference on High-Performance Graphics},
    year = {2012},
    pages = {33--37},
    numpages = {5},
    url = {https://doi.org/10.2312/EGGH/HPG12/033-037}, 
    doi = {10.2312/EGGH/HPG12/033-037},
    publisher = {Eurographics Association}
}
```

## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and
conditions](https://github.com/vchoutas/mesh-mesh-intersection/blob/master/LICENSE) and any
accompanying documentation before you download and/or use the SMPL-X/SMPLify-X
model, data and software, (the "Model & Software"), including 3D meshes, blend
weights, blend shapes, textures, software, scripts, and animations. By
downloading and/or using the Model & Software (including downloading, cloning,
installing, and any other use of this github repository), you acknowledge that
you have read these terms and conditions, understand them, and agree to be bound
by them. If you do not agree with these terms and conditions, you must not
download and/or use the Model & Software. Any infringement of the terms of this
agreement will automatically terminate your rights under this
[License](./LICENSE).




## Contact
The code of this repository was implemented by [Vassilis Choutas](vassilis.choutas@tuebingen.mpg.de).

For questions, please contact [smplx@tue.mpg.de](smplx@tue.mpg.de). 

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](ps-licensing@tue.mpg.de). Please note that the method for this component has been [patented by NVidia](https://patents.google.com/patent/US9396512B2/en) and a license needs to be obtained also by them.
