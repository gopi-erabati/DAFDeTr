import argparse
import os
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Results from Pickle File')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('results', help='Results pickle file')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # load results file
    results = mmcv.load(args.results)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                _module_path = cfg.plugin_dir
                # _module_dir = os.path.dirname(plugin_dir)
                # _module_dir = _module_dir.split('/')
                # _module_path = _module_dir[0]
                #
                # for m in _module_dir[1:]:
                #     _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    # Show the results using show() function in Dataset class
    dataset.evaluate(results)


if __name__ == '__main__':
    main()
