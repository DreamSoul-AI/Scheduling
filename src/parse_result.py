import argparse
import datetime
import json
import os
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from module import process_control

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
    cfg['logger_path'] = os.path.join('output', 'logger', 'train', 'runs', cfg['tag'])
    data = load_json_files(cfg['logger_path'])
    parse_data(data)
    return


def tree():
    return defaultdict(tree)


def parse_data(data):
    result = tree()
    for filename, data_i in data.items():
        worker_name = filename.split('.')[0]
        base_time = int(filename.split('.')[1]) / 1e9
        base_time = datetime.datetime.fromtimestamp(base_time)
        result[filename]['worker_name'] = worker_name
        result[filename]['base_time'] = base_time
        result[filename]['trace']['memory'] = []
        for i in range(len(data_i['traceEvents'])):
            trace_i = data_i['traceEvents'][i]
            parse_trace(trace_i, result[filename]['trace'])
    return result


def parse_trace(trace, result):
    if 'ProfilerStep' in trace['name']:
        step = int(trace['name'].split('#')[1])
        result['step'][step]['duration'] = trace['dur'] / 1e6
        result['step'][step]['ts'] = trace['ts'] / 1e9
    if trace['name'] == '[memory]':
        memory = {}
        if 'Total Allocated' in trace['args'] or 'Total Reserved' in trace['args']:
            memory['ts'] = trace['ts'] / 1e9
        if 'Total Allocated' in trace['args']:
            memory['Total Allocated'] = trace['args']['Total Allocated']
        if 'Total Reserved' in trace['args']:
            memory['Total Reserved'] = trace['args']['Total Reserved']
        result['memory'].append(memory)
    return


def load_json_files(logger_path):
    json_data = {}
    for filename in os.listdir(logger_path):
        if filename.endswith('.json'):
            with open(os.path.join(logger_path, filename), 'r') as f:
                json_data[filename] = json.load(f)
    return json_data


if __name__ == "__main__":
    main()
