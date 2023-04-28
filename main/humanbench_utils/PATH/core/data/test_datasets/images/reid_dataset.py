try:
    import mc
except:
    print("no mc")
from torch.utils.data import Dataset
import numpy as np
import random
import struct
import os
import cv2
import os.path as osp
import torch
import time
from core.utils import cv2_loader, pil_loader
from core.data.transforms.reid_transforms import ReidTestAugmentationCV2, ReidTestAugmentation, ReidTestAugmentationTimm
try:
    import spring.linklink as link
except:
    import linklink as link

from core.utils import sync_print
try:
    from petrel_client.client import Client as Client

    s3client = Client(boto=True,
                      enable_multi_cluster=True,
                      enable_mc = True)
except:
    print("ceph can not be used")

class ReIDTestDataset(Dataset):
    def __init__(self, image_list_paths, augmentation, root_path=None, vit=False, loader='cv2', **kwargs):
        self.root_path = root_path
        height = augmentation['height']
        width = augmentation['width']
        self.loader = loader
        if self.loader == 'cv2':
            self.transform = ReidTestAugmentationCV2(height, width, vit=vit)
        elif self.loader == 'pil':
            self.transform = ReidTestAugmentation(height, width, vit=vit)
        else:
            raise ValueError("{} loader is not supported!".format(self.loader))

        image_path = []; image_pid = []; image_camera = []
        pid_set = set()
        offset = 0
        for image_list_path in image_list_paths:
            print("building dataset from %s" % image_list_path)

            if image_list_path.startswith('/mnt/'):
                # image_list_path (image pid camid)
                with open(image_list_path, 'r') as f:
                    for line in f.readlines():
                        item = line.strip('\n').split(" ")
                        image_path.append(item[0])
                        pid = int(item[1])+offset
                        image_pid.append(pid)
                        pid_set.add(pid)
                        if len(item) >= 3:
                            image_camera.append(int(item[2]))
                            continue
                        # if not exist camera info, we use a fake data
                        image_camera.append(-1)
                # 777 is a magic number
                offset = max(image_pid) + 777
            elif 's3://' in image_list_path:
                # image_list_path (image pid camid)
                list_lines = s3client.Get(image_list_path)
                if not list_lines:
                    print('File not exist', list_file)
                    import pdb; pdb.set_trace()
                    raise IOError('File not exist', list_file)
                list_lines = list_lines.decode('ascii')
                list_lines = list_lines.split('\n')
                list_lines = list_lines[:-1]
                for line in list_lines:
                    item = line.strip('\n').split(" ")
                    image_path.append(item[0])
                    pid = int(item[1])+offset
                    image_pid.append(pid)
                    pid_set.add(pid)
                    if len(item) >= 3:
                        image_camera.append(int(item[2]))
                        continue
                    # if not exist camera info, we use a fake data
                    image_camera.append(-1)
                # 777 is a magic number
                offset = max(image_pid) + 777
            else:
                raise ValueError("Invalid Filename: {}".format(image_list_path))


        self.image_list = list(zip(image_path, image_pid, image_camera))
        self.image_list = sorted(self.image_list, key=lambda d: d[0])
        self.num_classes = len(pid_set)
        self.num_images = len(self.image_list)
        print("Successfully Loaded Data, Totally {} IDs".format(self.num_classes))
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            ## only use mc default
            print("==> will load files from local machine")
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.memcached_mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            ## mc-support-ceph
            print('mc-support-ceph')
            self.ceph_mclient = s3client

            self.initialized = True

    def get_num_classes(self):
        return self.num_classes

    def get_image_list(self):
        return self.image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, indices):
        self._init_memcached()
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _read_from_memcache(self, filename):
        if not osp.exists(filename):
            raise IOError('File is not exist:', filename)
        value = mc.pyvector()
        self.memcached_mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        if self.loader == 'cv2':
            img = cv2_loader(value_str)
        elif self.loader == 'pil':
            img = pil_loader(value_str)
        return img

    def _read_from_ceph(self, filename):
        value = self.ceph_mclient.Get(filename)
        if not value:
            raise FileNotFoundError("File is not exist on ceph: {}".format(filename))
        value_str = np.fromstring(value, np.uint8)
        if self.loader == 'cv2':
            img = cv2_loader(value)
        elif self.loader == 'pil':
            img = pil_loader(value_str)
        return img

    def _get_single_item(self, index=None):
        if index is None:
            index = np.random.randint(self.num_images)

        fname, pid, camid = self.image_list[index]
        filename = fname
        if self.root_path is not None:
            filename = osp.join(self.root_path, fname)
        try:
            if filename.startswith('/mnt/') or filename.startswith('/data/'):
                img = self._read_from_memcache(filename)
            elif 's3://' in filename:
                img = self._read_from_ceph(filename)
            else:
                raise KeyError('No such data source!')
        except BaseException:
            outputName = "failed_to_read_in_test.txt"
            with open(outputName, 'a') as g:
                g.write("%s\n"%(filename))
            print('Read test image[{}] failed ({})'.format(index, filename))
            raise IOError('File is not exist:', filename)
            return self._get_single_item()
        else:
            if self.transform is not None:
                img = self.transform(img)

        output = dict()
        output['image'] = img
        output['label'] = pid
        output['camera'] = camid
        output['index'] = index
        output['filename'] = fname
        return output

class ReIDTestDatasetDev(Dataset):
    def __init__(self, query_file_path, gallery_file_path, augmentation, root_path=None, vit=False, loader='cv2', **kwargs):
        self.root_path = root_path
        height = augmentation['height']
        width = augmentation['width']
        self.loader = loader
        if self.loader == 'cv2':
            self.transform = ReidTestAugmentationCV2(height, width, vit=vit)
        elif self.loader == 'pil':
            self.transform = ReidTestAugmentation(height, width, vit=vit)
        else:
            raise ValueError("{} loader is not supported!".format(self.loader))

        image_path = []; image_pid = []; image_camera = []; image_type = []
        pid_set = set()
        offset = 0
        for image_list_path in query_file_path:
            print("building dataset from %s" % image_list_path)

            if image_list_path.startswith('/mnt/'):
                # image_list_path (image pid camid)
                with open(image_list_path, 'r') as f:
                    for line in f.readlines():
                        item = line.strip('\n').split(" ")
                        image_path.append(item[0])
                        pid = int(item[1])+offset
                        image_pid.append(pid)
                        image_type.append('query')
                        pid_set.add(pid)
                        if len(item) >= 3:
                            image_camera.append(int(item[2]))
                            continue
                        # if not exist camera info, we use a fake data
                        image_camera.append(-1)
                # 777 is a magic number
                offset = max(image_pid) + 777
            elif 's3://' in image_list_path:
                # image_list_path (image pid camid)
                list_lines = s3client.Get(image_list_path)
                if not list_lines:
                    print('File not exist', list_file)
                    import pdb; pdb.set_trace()
                    raise IOError('File not exist', list_file)
                list_lines = list_lines.decode('ascii')
                list_lines = list_lines.split('\n')
                list_lines = list_lines[:-1]
                for line in list_lines:
                    item = line.strip('\n').split(" ")
                    image_path.append(item[0])
                    image_type.append('query')
                    pid = int(item[1])+offset
                    image_pid.append(pid)
                    pid_set.add(pid)
                    if len(item) >= 3:
                        image_camera.append(int(item[2]))
                        continue
                    # if not exist camera info, we use a fake data
                    image_camera.append(-1)
                # 777 is a magic number
                offset = max(image_pid) + 777
            else:
                raise ValueError("Invalid Filename: {}".format(image_list_path))

        offset = 0
        for image_list_path in gallery_file_path:
            print("building dataset from %s" % image_list_path)

            if image_list_path.startswith('/mnt/'):
                # image_list_path (image pid camid)
                with open(image_list_path, 'r') as f:
                    for line in f.readlines():
                        item = line.strip('\n').split(" ")
                        image_path.append(item[0])
                        pid = int(item[1])+offset
                        image_pid.append(pid)
                        image_type.append('query')
                        pid_set.add(pid)
                        if len(item) >= 3:
                            image_camera.append(int(item[2]))
                            continue
                        # if not exist camera info, we use a fake data
                        image_camera.append(-1)
                # 777 is a magic number
                offset = max(image_pid) + 777
            elif 's3://' in image_list_path:
                # image_list_path (image pid camid)
                list_lines = s3client.Get(image_list_path)
                if not list_lines:
                    print('File not exist', list_file)
                    import pdb; pdb.set_trace()
                    raise IOError('File not exist', list_file)
                list_lines = list_lines.decode('ascii')
                list_lines = list_lines.split('\n')
                list_lines = list_lines[:-1]
                for line in list_lines:
                    item = line.strip('\n').split(" ")
                    image_path.append(item[0])
                    image_type.append('gallery')
                    pid = int(item[1])+offset
                    image_pid.append(pid)
                    pid_set.add(pid)
                    if len(item) >= 3:
                        image_camera.append(int(item[2]))
                        continue
                    # if not exist camera info, we use a fake data
                    image_camera.append(-1)
                # 777 is a magic number
                offset = max(image_pid) + 777
            else:
                raise ValueError("Invalid Filename: {}".format(image_list_path))

        self.image_list = list(zip(image_path, image_pid, image_camera, image_type))
        # self.image_list = sorted(self.image_list, key=lambda d: d[0])
        self.num_classes = len(pid_set)
        self.num_images = len(self.image_list)
        print("Successfully Loaded Data, Totally {} IDs".format(self.num_classes))
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            ## only use mc default
            print("==> will load files from local machine")
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.memcached_mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            ## mc-support-ceph
            print('mc-support-ceph')
            self.ceph_mclient = s3client

            self.initialized = True

    def get_num_classes(self):
        return self.num_classes

    def get_image_list(self):
        return self.image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, indices):
        self._init_memcached()
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _read_from_memcache(self, filename):
        if not osp.exists(filename):
            raise IOError('File is not exist:', filename)
        value = mc.pyvector()
        self.memcached_mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        if self.loader == 'cv2':
            img = cv2_loader(value_str)
        elif self.loader == 'pil':
            img = pil_loader(value_str)
        return img

    def _read_from_ceph(self, filename):
        value = self.ceph_mclient.Get(filename)
        if not value:
            raise FileNotFoundError("File is not exist on ceph: {}".format(filename))
        value_str = np.fromstring(value, np.uint8)
        if self.loader == 'cv2':
            img = cv2_loader(value)
        elif self.loader == 'pil':
            img = pil_loader(value_str)
        return img

    def _get_single_item(self, index=None):
        if index is None:
            index = np.random.randint(self.num_images)

        fname, pid, camid, ftype = self.image_list[index]
        filename = fname
        if self.root_path is not None:
            filename = osp.join(self.root_path, fname)
        try:
            if filename.startswith('/mnt/') or filename.startswith('/data/'):
                img = self._read_from_memcache(filename)
            elif 's3://' in filename:
                img = self._read_from_ceph(filename)
            else:
                raise KeyError('No such data source!')
        except BaseException:
            outputName = "failed_to_read_in_test.txt"
            with open(outputName, 'a') as g:
                g.write("%s\n"%(filename))
            print('Read test image[{}] failed ({})'.format(index, filename))
            raise IOError('File is not exist:', filename)
            return self._get_single_item()
        else:
            if self.transform is not None:
                img = self.transform(img)

        output = dict()
        output['image'] = img
        output['label'] = pid
        output['camera'] = camid
        output['index'] = index
        output['filename'] = fname
        output['image_type'] = ftype
        return output
