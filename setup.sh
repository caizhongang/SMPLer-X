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
wget http://download.openmmlab.sensetime.com/mmcv/dist/cu113/torch1.12.0/mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl
pip install mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl
rm mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl
pip install -r requirements.txt

# install mmpose
cd main/transformer_utils
pip install -v -e .
cd ../..

# add soft links
ln -s /mnt/lustrenew/share_data/zoetrope/osx/data dataset
ln -s /mnt/cache/share_data/zoetrope/body_models common/utils/human_model_files
ln -s /mnt/cache/share_data/zoetrope/osx/pretrained_models/osx_vit_l.pth pretrained_models/osx_vit_l.pth
ln -s /mnt/cache/share_data/zoetrope/osx/pretrained_models/osx_vit_b.pth pretrained_models/osx_vit_b.pth
ln -s /mnt/cache/share_data/zoetrope/osx/pretrained_models/osx_l.pth.tar pretrained_models/osx_l.pth.tar
ln -s /mnt/cache/share_data/zoetrope/osx/pretrained_models/osx_l_agora.pth.tar pretrained_models/osx_l_agora.pth.tar