import argparse
import datetime
import json
import os
import numpy as np
import matplotlib.pyplot as plt
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
    cfg['logger_path'] = os.path.join(cfg['tag_path'], 'logger', 'train')
    data = load_json_files(cfg['logger_path'])
    result = parse_data(data)
    for filename in result:
        print(result[filename]['worker_name'])
        print(result[filename]['base_time'])
        print(result[filename]['stats'])

        # print(result[filename]['trace']['step'])
        # memory = result[filename]['trace']['memory']
        # ts, total_allocated, total_reserved = [], [], []
        # for m in memory:
        #     ts.append(m['ts'])
        #     total_allocated.append(m.get('Total Allocated', 0))
        #     total_reserved.append(m.get('Total Reserved', 0))
        # # Sort based on ts and get sorted indices
        # sorted_indices = sorted(range(len(ts)), key=lambda i: ts[i])
        #
        # # Reorder the lists based on sorted indices
        # ts = [ts[i] for i in sorted_indices]
        # total_allocated = [total_allocated[i] for i in sorted_indices]
        # total_reserved = [total_reserved[i] for i in sorted_indices]
        # plt.figure()
        # # plt.plot(ts, total_allocated, label='Total Allocated')
        # plt.plot(ts, total_reserved, label='Total Reserved')
        # plt.xlabel('Timestamp (s)')
        # plt.ylabel('Memory (bytes)')
        # plt.title(f'Memory Usage for {filename}')
        # plt.legend()
        # plt.show()
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
        result[filename]['trace']['memory'] = {'ts': [], 'total_allocated': [], 'total_reserved': []}
        for i in range(len(data_i['traceEvents'])):
            trace_i = data_i['traceEvents'][i]
            parse_trace(trace_i, result[filename]['trace'])
        result[filename]['runtime'] = {'duration': [], 'ts': []}
        for step in result[filename]['trace']['step']:
            result[filename]['runtime']['duration'].append(result[filename]['trace']['step'][step]['duration'])
            result[filename]['runtime']['ts'].append(result[filename]['trace']['step'][step]['ts'])
        result[filename]['stats']['runtime'] = make_stats(result[filename]['runtime']['duration'])
        result[filename]['stats']['memory']['allocated'] = make_stats(
            result[filename]['trace']['memory']['total_allocated'])
        result[filename]['stats']['memory']['reserved'] = make_stats(
            result[filename]['trace']['memory']['total_reserved'])
    return result


def parse_trace(trace, result):
    if 'ProfilerStep' in trace['name']:
        step = int(trace['name'].split('#')[1])
        result['step'][step]['duration'] = trace['dur'] / 1e6
        result['step'][step]['ts'] = trace['ts'] / 1e9
    if trace['name'] == '[memory]':
        ts = trace['ts'] / 1e9
        total_allocated = trace['args']['Total Allocated']
        total_reserved = trace['args']['Total Reserved']
        result['memory']['ts'].append(ts)
        result['memory']['total_allocated'].append(total_allocated)
        result['memory']['total_reserved'].append(total_reserved)
    return


def load_json_files(logger_path):
    json_data = {}
    for filename in os.listdir(logger_path):
        if filename.endswith('.json'):
            with open(os.path.join(logger_path, filename), 'r') as f:
                json_data[filename] = json.load(f)
    return json_data


def make_stats(data):
    data = np.array(data)
    stats = {'mean': data.mean(), 'std': data.std(), 'max': (data.max(), data.argmax()),
             'min': (data.min(), data.argmin())}
    return stats


if __name__ == "__main__":
    main()
