import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from dataset import make_dataset, process_dataset
from config import cfg, process_args
from model import make_model
from module import process_control, save
from summarizer import Summarizer

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best')
    cfg['logger_path'] = os.path.join(cfg['tag_path'], 'logger', 'train')
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    model = make_model(cfg['model'])
    summarizer = Summarizer()
    result = summarizer.summarize(dataset, model)
    print(result)
    save(result, os.path.join(cfg['tag_path'], 'summary'))
    return

if __name__ == "__main__":
    main()