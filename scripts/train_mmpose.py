import click 
import os
from datetime import datetime

sbatch_script = """#!/bin/env bash
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem={mem}G
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -t {time}
#SBATCH --output={slurm_out}

source $HOME/.bashrc
module load gcc/9.2.0
module load cuda/11.7
conda activate /n/groups/datta/tim_sainburg/conda_envs/mmdeploy
python {pdir}/tmp_pose.py
"""

py_script = """
import os
from pathlib import Path
import sys
from datetime import datetime

from mmengine import Config
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model_name = '{model_name}'

# Where the COCO format dataset is located (created in the previous notebook)
dataset_directory = Path('{dataset_directory}')

# which config to use (this is what we base the config off of). Should be in the mmdeteciton repo. 
config_loc = Path('{config}')

pretrained_model = '{pretrained_model}'
if len(pretrained_model) > 0:
    pretrained_model = Path(pretrained_model)
    use_pretrained_model = True

# working directory (where model output is saved)
output_directory = Path('{output_directory}')
working_directory = (output_directory / 'rtmpose' / '{model_name}_{formatted_datetime}')
working_directory.mkdir(parents=True, exist_ok=True)


cfg = Config.fromfile(config_loc.as_posix())

# set the dataset directory
cfg.data_root = dataset_directory.as_posix()

# set the working directory
cfg.work_dir = working_directory.as_posix()

cfg.randomness = dict(seed=0)

if use_pretrained_model:
    cfg.load_from = pretrained_model.as_posix()

# set the metainfo
cfg.metainfo = {{
    'classes': ('Mouse', ),
    'palette': [
        (220, 20, 60),
    ]
}}

# set dataset configs
cfg.dataset_type = 'CoCo25pt'
cfg.data_mode = 'topdown'

# number of keypoints
cfg.model.head.out_channels = 25

cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.ann_file = 'annotations/instances_train.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img='train/')

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.bbox_file = None
cfg.val_dataloader.dataset.ann_file = 'annotations/instances_val.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img='val/')

cfg.test_dataloader.dataset.type = cfg.dataset_type
cfg.test_dataloader.dataset.bbox_file = None
cfg.test_dataloader.dataset.ann_file = 'annotations/instances_val.json'
cfg.test_dataloader.dataset.data_root = cfg.data_root
cfg.test_dataloader.dataset.data_prefix = dict(img='val/')

# set to custom datset
cfg.train_dataloader.dataset.metainfo = dict(from_file=dataset_info_loc.as_posix())
cfg.val_dataloader.dataset.metainfo = dict(from_file=dataset_info_loc.as_posix())
cfg.test_dataloader.dataset.metainfo = dict(from_file=dataset_info_loc.as_posix())

# set evaluator
cfg.val_evaluator = dict(type='PCKAccuracy')
cfg.test_evaluator = cfg.val_evaluator

cfg.default_hooks.checkpoint.save_best = 'PCK'
cfg.default_hooks.checkpoint.max_keep_ckpts = 15
cfg.default_hooks.checkpoint.interval = {ckpt_interval}

cfg.max_epochs = 2000
cfg.train_cfg.max_epochs = 2000

# set preprocess configs to model
cfg.model.setdefault('data_preprocessor', cfg.get('preprocess_cfg', {}))

# save configuration file for future reference
cfg.dump(working_directory / 'config.py')

# build the runner from config
runner = Runner.from_cfg(cfg)

# start training
runner.train()
"""

@click.command()
@click.argument('model_name', type=str)
@click.argument('dataset_directory', type=str)
@click.argument('config', type=str)
@click.option('--pretrained_model', default=None, type=str)
@click.option('--output_directory', default='.', type=str)
@click.option('--ckpt_interval', default=50, type=int)
@click.option('--mem', default=16, type=int)
@click.option('--time', default='24:00:00', type=str)
@click.option('--slurm_out', default='train_mmdet.out', type=str)
@click.option('--pdir', default='.', type=str)
@click.option('--dry_run', is_flag=True)
def main(model_name, dataset_directory, config, pretrained_model, output_directory, ckpt_interval, mem, time, slurm_out, pdir, dry_run):
    print(f"model_name: {model_name}")
    _py = py_script.format(
        model_name=model_name, 
        dataset_directory=dataset_directory, 
        config=config,
        pretrained_model=pretrained_model,
        formatted_datetime = datetime.now().strftime("%y-%m-%d-%H-%M-%S"), 
        output_directory=output_directory,
        ckpt_interval=ckpt_interval)
    
    with open(pdir + '/tmp_pose.py', 'w') as f:
        f.write(_py)
    
    _sbatch = sbatch_script.format(
        mem=mem, 
        time=time, 
        slurm_out=slurm_out, 
        pdir=pdir)
    
    with open(pdir + '/train_mmpose.sh', 'w') as f:
        f.write(_sbatch)

    if not dry_run:
        os.system(f'sbatch {pdir}/train_mmpose.sh')

if __name__ == "__main__":
    main()