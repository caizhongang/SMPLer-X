# Setup script on 1988
# Note you have to setup conda and pip sources first

# set gcc environment
export CXX=/mnt/lustre/share/gcc-7.4.0/bin/g++
export CC=/mnt/lustre/share/gcc-7.4.0/bin/gcc
export GCC=/mnt/lustre/share/gcc-7.4.0/bin/gcc
export PATH=/mnt/lustre/share/gcc-7.4.0/bin:/mnt/lustre/share/gcc-7.4.0/install/gcc-7.4.0/lib64:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc-7.4.0/lib:$LD_LIBRARY_PATH

# set cuda environment
export PATH=/mnt/lustre/share/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-10.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/lustre/share/cuda-10.2

# create conda environment
conda create -n osx python=3.8 -y
conda activate osx

# install packages
conda install pytorch=1.8.0 torchvision cudatoolkit=10.2 -c pytorch -y
pip install "mmcv-full==1.5.0" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
pip install -r requirements.txt  # removed mmcv-full version
pip install timm

# install mmpose
cd main/transformer_utils
pip install -v -e .
cd ../..

# add soft links
ln -s /mnt/lustrenew/share_data/zoetrope/osx/data dataset
ln -s /mnt/cache/share_data/zoetrope/body_models common/utils/human_model_files
ln -s /mnt/cache/share_data/zoetrope/osx/pretrained_models/osx_vit_l.pth osx_vit_l.pth