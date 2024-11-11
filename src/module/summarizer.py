import copy
import gc
import numpy as np
import torch
from functools import partial
from .hardware import Monitor
from .utils import to_device
from ..dataset import make_data_loader


class Summarizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = 1
        self.monitor = Monitor(copy.deepcopy(self.cfg))

    def summarize(self, dataset, model, data_collator=None):
        summary = {}
        summary['data'] = self.make_data(dataset)
        summary['hardware'] = self.monitor.hardware()
        summary['params'] = self.make_params(model)
        summary['module_names'] = self.make_module_names(model, self.cfg['summarize']['finetune'])
        summary['param_names'] = self.make_param_names(model, summary['module_names'],
                                                       self.cfg['summarize']['target_param_types'])
        summary['buffer_names'] = self.make_buffer_names(model)
        if 'train' in dataset:
            model_summary = self.make_model_summary('train', dataset, model, summary['module_names'],
                                                    summary['param_names'], summary['buffer_names'], data_collator)
            summary['batch_size'] = model_summary['batch_size']
            summary['module_names_forward'] = model_summary['module_names_forward']
            summary['module_names_backward'] = model_summary['module_names_backward']
            summary['param_names_backward'] = model_summary['param_names_backward']
            summary['activation'] = model_summary['activation']
            summary['activation_offset'] = model_summary['activation_offset']
            summary['tied_param_names'] = model_summary['tied_param_names']
        else:
            self.preload_model(model)
            model_summary = self.make_model_summary('test', dataset, model, summary['module_names'],
                                                    summary['param_names'], data_collator)
            summary['batch_size'] = model_summary['batch_size']
            summary['module_names_forward'] = model_summary['module_names_forward']

        summary['hardware'] = self.monitor.hardware()
        summary['safety'] = self.make_safety(summary['params'], summary['hardware'])
        return summary

    def make_data(self, dataset):
        data_info = {}
        for k in dataset:
            data_info[k] = {'data_size': len(dataset[k]), 'date_shape': dataset[k].data_shape,
                            'target_size': dataset[k].target_size}
        return data_info

    def make_params(self, model):
        params = {'total': {}, 'trainable': {}}
        total_num_params = 0
        total_param_memory = 0
        trainable_num_params = 0
        trainable_param_memory = 0
        for param in model.parameters():
            _, _, count, memory = make_stats(param)
            total_num_params += count
            total_param_memory += memory
            if param.requires_grad:
                trainable_num_params += count
                trainable_param_memory += memory
        params['total']['count'] = total_num_params
        params['total']['memory'] = total_param_memory
        params['trainable']['count'] = trainable_num_params
        params['trainable']['memory'] = trainable_param_memory
        return params

    def make_safety(self, params, status):
        safety = {k: {} for k in status}
        safety['cpu']['memory'] = (1 - self.cfg['hardware']['cpu']['safety']) * status['cpu']['free_memory']
        safety['available'] = [self.cfg['hardware']['available'][0]]
        if self.cfg['hardware']['cuda']['is_available']:
            safety['cuda']['memory'] = []
            for i in range(len(status['cuda'])):
                safety['cuda']['memory'].append((1 - self.cfg['hardware']['cuda']['safety']) *
                                                status['cuda'][i]['free_memory'])
                if safety['cuda']['memory'][i] < params['total']['memory']:
                    print('Model cannot fit in memory of cuda:{}'.format(i))
                    self.cfg['hardware']['available'].remove('cuda:{}'.format(i))
            if len(self.cfg['hardware']['available']) < self.cfg['dist']['num_devices'] + 1:
                raise ValueError('Not enough cuda devices can fit the model')

            sorted_cuda_memory = np.argsort(safety['cuda']['memory'])[::-1].tolist()
            available_sorted_cuda_memory = [self.cfg['hardware']['available'][1:][idx] for idx in sorted_cuda_memory]
            safety['available'] = safety['available'] + available_sorted_cuda_memory
        return safety

    def make_module_names(self, model, finetune=None):
        target_module_names = []
        for module_name, module in model.named_modules():
            if any(p.numel() > 0 for p in module.parameters(recurse=False)) or \
                    any(p.numel() > 0 for p in module.buffers(recurse=False)):
                target_module_names.append(module_name)

        module_names = []
        for module_name, module in model.named_modules():
            if any(module_name.endswith(target_module_name) for target_module_name in target_module_names):
                module_names.append(module_name)
        return module_names

    def make_param_names(self, model, module_names, target_param_types='all'):
        def filter_param(param_name_, param_):
            if target_param_types == 'all':
                if not any(module_name in param_name_ for module_name in module_names):
                    param_.requires_grad = False
            elif isinstance(target_param_types, list) and \
                    (any(param_type in param_name_ for param_type in target_param_types)):
                pass
            else:
                param_.requires_grad = False
            return

        param_names = {}
        for module_name, module in model.named_modules():
            if any(p.numel() > 0 for p in module.parameters(recurse=False)):
                for param_name, param in module.named_parameters(recurse=False):
                    full_param_name = '{}.{}'.format(module_name, param_name)
                    filter_param(full_param_name, param)
                    size, dtype, count, memory = make_stats(param)
                    param_names[full_param_name] = {'size': size, 'dtype': dtype, 'count': count, 'memory': memory,
                                                    'requires_grad': param.requires_grad}
        return param_names

    def make_buffer_names(self, model):
        buffer_names = {}
        for module_name, module in model.named_modules():
            if any(p.numel() > 0 for p in module.buffers(recurse=False)):
                for buffer_name, buffer in model.named_buffers(recurse=False):
                    full_buffer_name = '{}.{}'.format(module_name, buffer_name)
                    size, dtype, count, memory = make_stats(buffer)
                    buffer_names[full_buffer_name] = {'size': size, 'dtype': dtype, 'count': count, 'memory': memory}
        return buffer_names

    def make_model_summary(self, mode, dataset, model, module_names, param_names, buffer_names, data_collator=None):
        def forward_hook(module_name, has_parameters, has_buffers, module, args, output):
            module_names_forward[module_name] = {'param': {}, 'buffer': {}}
            if has_parameters:
                for param_name, param in module.named_parameters():
                    param_name = '{}.{}'.format(module_name, param_name)
                    size, dtype, count, memory = make_stats(param)
                    module_names_forward[module_name]['param'][param_name] = {'size': size, 'dtype': dtype,
                                                                              'count': count, 'memory': memory}
            if has_buffers:
                for buffer_name, buffer in module.named_buffers():
                    buffer_name = '{}.{}'.format(module_name, buffer_name)
                    size, dtype, count, memory = make_stats(buffer)
                    module_names_forward[module_name]['buffer'][buffer_name] = {'size': size, 'dtype': dtype,
                                                                                'count': count, 'memory': memory}
            return

        def backward_hook(module_name, has_parameters, has_buffers, module, grad_input, grad_output):
            module_names_backward[module_name] = {'param': {}, 'buffer': {}}
            if has_parameters:
                for param_name, param in module.named_parameters():
                    param_name = '{}.{}'.format(module_name, param_name)
                    size, dtype, count, memory = make_stats(param)
                    module_names_backward[module_name]['param'][param_name] = {'size': size, 'dtype': dtype,
                                                                               'count': count, 'memory': memory}
            if has_buffers:
                for buffer_name, buffer in module.named_buffers():
                    buffer_name = '{}.{}'.format(module_name, buffer_name)
                    size, dtype, count, memory = make_stats(buffer)
                    module_names_backward[module_name]['buffer'][buffer_name] = {'size': size, 'dtype': dtype,
                                                                                 'count': count, 'memory': memory}
            return

        def grad_hook(summary, param_name, param):
            param.grad = None
            size, dtype, count, memory = make_stats(param)
            summary[param_name] = {'size': size, 'dtype': dtype, 'count': count, 'memory': memory, 'tied': 0}
            return

        def clean_hook():
            for param_hook_i in param_hook:
                param_hook_i.remove()
            for modules_names_forward_hook_i in modules_names_forward_hook:
                modules_names_forward_hook_i.remove()
            for module_names_backward_hook_i in module_names_backward_hook:
                module_names_backward_hook_i.remove()
            return

        original_device = next(iter(model.parameters())).device
        if str(original_device) != 'cpu':
            msg = 'Original device on {}, not on cpu'.format(original_device)
            print(msg)
        if self.cfg['hardware']['cuda']['is_available']:
            map_device = self.cfg['hardware']['available'][1]
        else:
            map_device = original_device
        model = model.to(map_device)
        orig_buffer = {}
        for buffer_name, buffer in model.named_buffers():
            orig_buffer[buffer_name] = copy.deepcopy(buffer)

        module_names_forward = {}
        modules_names_forward_hook = []
        module_names_backward = {}
        module_names_backward_hook = []

        for module_name, module in model.named_modules():
            has_parameters = any(p.numel() > 0 for p in module.parameters(recurse=False))
            has_buffers = any(p.numel() > 0 for p in module.buffers(recurse=False))
            if has_parameters or has_buffers:
                modules_names_forward_hook_i = module.register_forward_hook(
                    partial(forward_hook, module_name, has_parameters, has_buffers))
                modules_names_forward_hook.append(modules_names_forward_hook_i)
                if mode == 'learn':
                    module_names_backward_hook_i = module.register_full_backward_hook(partial(
                        backward_hook, module_name, has_parameters, has_buffers))
                    module_names_backward_hook.append(module_names_backward_hook_i)

        param_names_backward = {}
        param_hook = []
        if mode == 'learn':
            for param_name, param in model.named_parameters():
                if param_name in param_names and param.requires_grad:
                    param_hook.append(param.register_post_accumulate_grad_hook(
                        partial(grad_hook, param_names_backward, param_name)))

        tied_param_names = {}
        for module_name, module in model.named_modules():
            if any(p.numel() > 0 for p in module.parameters(recurse=False)):
                for param_name, param in module.named_parameters(recurse=False):
                    full_param_name = '{}.{}'.format(module_name, param_name)
                    for unique_param_name, unique_param in model.named_parameters():
                        if full_param_name != unique_param_name and id(param) == id(unique_param):
                            tied_param_names[full_param_name] = unique_param_name
            if any(p.numel() > 0 for p in module.buffers(recurse=False)):
                for buffer_name, buffer in module.named_buffers(recurse=False):
                    full_buffer_name = '{}.{}'.format(module_name, buffer_name)
                    for unique_buffer_name, unique_buffer in model.named_buffers():
                        if full_buffer_name != unique_buffer_name and id(buffer) == id(unique_buffer):
                            tied_param_names[full_buffer_name] = unique_buffer_name

        activation = {}
        activation_offset = {}
        self.run(mode, dataset, model, self.batch_size, map_device, activation, data_collator)
        clean_hook()
        if mode == 'learn':
            self.run(mode, dataset, model, self.batch_size + 1, map_device, activation_offset, data_collator)
        model.train(False)
        model.to(original_device)
        for buffer_name, buffer in model.named_buffers():
            buffer.data.copy_(orig_buffer[buffer_name].data)
        self.clean()

        result = {'batch_size': self.batch_size, 'param_names_backward': param_names_backward, 'activation': activation,
                  'activation_offset': activation_offset, 'module_names_forward': module_names_forward,
                  'module_names_backward': module_names_backward, 'tied_param_names': tied_param_names}
        return result

    def preload_model(self, model):
        original_device = next(iter(model.parameters())).device
        if str(original_device) != 'cpu':
            msg = 'Original device on {}, not on cpu'.format(original_device)
            print(msg)
        if self.cfg['hardware']['cuda']['is_available'] and self.cfg['col']['device']['upload'] == 'cuda':
            map_device = self.cfg['hardware']['available'][1]
        else:
            map_device = original_device
        model = model.to(map_device)
        model.to(original_device)
        self.clean()
        return

    def run(self, mode, dataset, model, batch_size, device, activation, data_collator=None):
        if mode == 'learn':
            model.train(True)
            index_tracker = 0
            self.cfg['learn']['batch_size'] = batch_size
            data_loader = make_data_loader(dataset, data_collator, **self.cfg)
            split = list(data_loader['learn'].keys())[0]
            input = next(iter(data_loader['learn'][split]))
            input = to_device(input, device)
            with summarize_activation(activation, index_tracker):
                output = model(**input)
            output['loss'].backward()
            model.zero_grad()
        else:
            model.train(False)
            self.cfg['eval']['batch_size'] = batch_size
            data_loader = make_data_loader(dataset, data_collator, **self.cfg)
            split = list(data_loader['eval'].keys())[0]
            input = next(iter(data_loader['eval'][split]))
            input = to_device(input, device)
            with torch.no_grad():
                output = model(**input)
        return

    def clean(self):
        gc.collect()
        if self.cfg['hardware']['cuda']['is_available']:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return


class summarize_activation(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, summary, index_tracker):
        self.index_tracker = index_tracker

        def pack(tensor):
            size, dtype, count, memory = make_stats(tensor)
            summary[self.index_tracker] = {'size': size, 'dtype': dtype, 'count': count, 'memory': memory}
            self.index_tracker += 1
            return tensor

        def unpack(tensor):
            return tensor

        super().__init__(pack, unpack)


def make_stats(input):
    size = input.size()
    dtype = input.dtype
    count = input.numel()
    element_size = input.element_size()
    memory = count * element_size / 1024 ** 2
    return size, dtype, count, memory
