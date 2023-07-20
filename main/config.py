import os
import os.path as osp
import sys
import datetime
from mmcv import Config as MMConfig

class Config:
    def get_config_fromfile(self, config_path):
        self.config_path = config_path
        cfg = MMConfig.fromfile(self.config_path)
        self.__dict__.update(dict(cfg))

        # update dir
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        self.root_dir = osp.join(self.cur_dir, '..')
        self.data_dir = osp.join(self.root_dir, 'dataset')
        self.human_model_path = osp.join(self.root_dir, 'common', 'utils', 'human_model_files')

        ## add some paths to the system root dir
        sys.path.insert(0, osp.join(self.root_dir, 'common'))
        from utils.dir import add_pypath
        add_pypath(osp.join(self.data_dir))
        for dataset in os.listdir(osp.join(self.root_dir, 'data')):
            if dataset not in ['humandata.py', '__pycache__', 'dataset.py']:
                add_pypath(osp.join(self.root_dir, 'data', dataset))
        add_pypath(osp.join(self.root_dir, 'data'))
        add_pypath(self.data_dir)
                
    def prepare_dirs(self, exp_name):
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = osp.join(self.root_dir, f'{exp_name}_{time_str}')
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.code_dir = osp.join(self.output_dir, 'code')
        self.result_dir = osp.join(self.output_dir, 'result')

        from utils.dir import make_folder
        make_folder(self.model_dir)
        make_folder(self.vis_dir)
        make_folder(self.log_dir)
        make_folder(self.code_dir)
        make_folder(self.result_dir)

        ## copy some code to log dir as a backup
        copy_files = ['main/train.py', 'main/test.py', 'common/base.py',
                      'common/nets', 'main/SMPLer_X.py',
                      'data/dataset.py', 'data/MSCOCO/MSCOCO.py', 'data/AGORA/AGORA.py']
        for file in copy_files:
            os.system(f'cp -r {self.root_dir}/{file} {self.code_dir}')

    def update_test_config(self, testset, agora_benchmark, shapy_eval_split, pretrained_model_path, use_cache,
                           eval_on_train=False, vis=False):
        self.testset = testset
        self.agora_benchmark = agora_benchmark
        self.pretrained_model_path = pretrained_model_path
        self.shapy_eval_split = shapy_eval_split
        self.use_cache = use_cache
        self.eval_on_train = eval_on_train
        self.vis = vis

    def update_config(self, num_gpus, exp_name):
        self.num_gpus = num_gpus
        self.exp_name = exp_name
        
        self.prepare_dirs(self.exp_name)
        
        # Save
        cfg_save = MMConfig(self.__dict__)
        cfg_save.dump(osp.join(self.code_dir,'config_base.py'))

cfg = Config()