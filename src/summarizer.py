import copy
import torch
from functools import partial
from module.hardware import Monitor
from module.utils import to_device
from dataset import make_data_loader


class Summarizer:
    def __init__(self):
        self.batch_size = 1
        self.monitor = Monitor()

    def summarize(self, dataset, model):
        summary = {}
        summary['data'] = self.make_data(dataset)
        summary['hardware'] = self.monitor.hardware()
        summary['params'] = self.make_params(model)
        if 'train' in dataset:
            model_summary = self.make_model_summary('train', dataset, model)
            summary['batch_size'] = model_summary['batch_size']
            summary['module_names_forward'] = model_summary['module_names_forward']
            summary['module_names_backward'] = model_summary['module_names_backward']
            summary['param_names_backward'] = model_summary['param_names_backward']
            summary['activation'] = model_summary['activation']
            summary['activation_offset'] = model_summary['activation_offset']
            summary['tied_param_names'] = model_summary['tied_param_names']
        else:
            self.preload_model(model)
            model_summary = self.make_model_summary('test', dataset, model)
            summary['batch_size'] = model_summary['batch_size']
            summary['module_names_forward'] = model_summary['module_names_forward']

        summary['hardware'] = self.monitor.hardware()
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

    def make_model_summary(self, mode, dataset, model):
        def forward_hook(module_name, has_parameters, has_buffers, tied_param_names, module_index, param_index, buffer_index, module, args, output): ## TODO: continue fix this
            direction = 'forward'
            name = 'module_{}_{}_{}'.format(param_index[0], direction, module_name)
            info = {'type': 'module', 'index': param_index[0], 'direction': direction, 'name': module_name}
            size, dtype, count, memory = [], [], [], []
            if has_parameters:
                for param_name, param in module.named_parameters(recurse=False):
                    full_param_name = '{}.{}'.format(module_name, param_name)
                    if full_param_name in tied_param_names:
                        full_param_name = tied_param_names[full_param_name]
                    name_i = 'param_{}_{}_{}'.format(param_index[0], direction, module_name)
                    size, dtype, count, memory = make_stats(param)
                    module_names_forward[module_name]['param'][param_name] = {'size': size, 'dtype': dtype,
                                                                              'count': count, 'memory': memory}
            if has_buffers:
                for buffer_name, buffer in module.named_buffers(recurse=False):
                    buffer_name = '{}.{}'.format(module_name, buffer_name)
                    size, dtype, count, memory = make_stats(buffer)
                    module_names_forward[module_name]['buffer'][buffer_name] = {'size': size, 'dtype': dtype,
                                                                                'count': count, 'memory': memory}
            stats = {'size': size, 'dtype': dtype, 'count': count, 'memory': memory}
            summary[name] = {'info': info, 'stats': stats}
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

        def grad_hook(summary, grad_index, param_name, param):
            param.grad = None
            direction = 'backward'
            name = 'grad_{}_{}_{}'.format(grad_index[0], direction, param_name)
            info = {'type': 'grad', 'index': grad_index[0], 'direction': direction, 'name': param_name}
            size, dtype, count, memory = make_stats(param)
            stats = {'size': size, 'dtype': dtype, 'count': count, 'memory': memory}
            summary[name] = {'info': info, 'stats': stats}
            grad_index[0] += 1
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
        if torch.cuda.is_available:
            map_device = 'cuda'
        else:
            map_device = original_device
        model = model.to(map_device)
        orig_buffer = {}
        for buffer_name, buffer in model.named_buffers():
            orig_buffer[buffer_name] = copy.deepcopy(buffer)
        summary = {}

        param_hook = []
        grad_index = [0]
        if mode == 'train':
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    param_hook.append(param.register_post_accumulate_grad_hook(
                        partial(grad_hook, summary, grad_index, param_name)))

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

        modules_names_forward_hook = []
        module_names_backward_hook = []
        param_index = [0]
        for module_name, module in model.named_modules():
            has_parameters = any(p.numel() > 0 for p in module.parameters(recurse=False))
            has_buffers = any(p.numel() > 0 for p in module.buffers(recurse=False))
            modules_names_forward_hook_i = module.register_forward_hook(
                partial(forward_hook, module_name, has_parameters, has_buffers, tied_param_names, param_index))
            modules_names_forward_hook.append(modules_names_forward_hook_i)
            if mode == 'train':
                module_names_backward_hook_i = module.register_full_backward_hook(partial(
                    backward_hook, module_name, has_parameters, has_buffers, tied_param_names, param_index))
                module_names_backward_hook.append(module_names_backward_hook_i)

        self.run(mode, dataset, model, self.batch_size, map_device, summary, offset=False)
        clean_hook()
        if mode == 'train':
            self.run(mode, dataset, model, self.batch_size + 1, map_device, summary, offset=True)
        model.train(False)
        model.to(original_device)
        for buffer_name, buffer in model.named_buffers():
            buffer.data.copy_(orig_buffer[buffer_name].data)
        for name in summary:
            print(name, summary[name])
        exit()
        # result = {'batch_size': self.batch_size, 'param_names_backward': param_names_backward, 'activation': activation,
        #           'activation_offset': activation_offset, 'module_names_forward': module_names_forward,
        #           'module_names_backward': module_names_backward, 'tied_param_names': tied_param_names}
        result = {'batch_size': self.batch_size, 'summary': summary}
        return result

    def preload_model(self, model):
        original_device = next(iter(model.parameters())).device
        if str(original_device) != 'cpu':
            msg = 'Original device on {}, not on cpu'.format(original_device)
            print(msg)
        if torch.cuda.is_available:
            map_device = 'cuda'
        else:
            map_device = original_device
        model = model.to(map_device)
        model.to(original_device)
        return

    def run(self, mode, dataset, model, batch_size, device, summary, offset):
        if mode == 'train':
            model.train(True)
            index_tracker = 0
            activation_index = 0
            data_loader = make_data_loader({'train': dataset['train']}, batch_size={'train': batch_size})
            input = next(iter(data_loader['train']))
            input = to_device(input, device)
            with summarize_activation(summary, offset, activation_index, index_tracker):
                output = model(**input)
            output['loss'].backward()
            model.zero_grad()
        else:
            model.train(False)
            data_loader = make_data_loader({'test': dataset['test']}, batch_size={'test': batch_size})
            input = next(iter(data_loader['test']))
            input = to_device(input, device)
            with torch.no_grad():
                output = model(**input)
        return


class summarize_activation(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, summary, offset, activation_index, index_tracker):
        self.activation_index = activation_index
        self.index_tracker = index_tracker

        def pack(tensor):
            direction = 'forward'
            name = 'activation_{}_{}_{}'.format(self.activation_index, direction, self.index_tracker)
            info = {'type': 'activation', 'index': self.activation_index, 'direction': direction,
                    'name': self.index_tracker}
            size, dtype, count, memory = make_stats(tensor)
            stats = {'size': size, 'dtype': dtype, 'count': count, 'memory': memory}
            if offset and name in summary:
                info['batch_factor'] = [size[i] / summary[name]['stats']['size'][i] for i in range(len(size))]
            else:
                info['batch_factor'] = None
            summary[name] = {'info': info, 'stats': stats}
            self.activation_index += 1
            self.index_tracker += 1
            return (tensor, self.index_tracker - 1)

        def unpack(input):
            tensor, index = input
            direction = 'backward'
            name = 'activation_{}_{}_{}'.format(self.activation_index, index, direction)
            info = {'type': 'activation', 'index': self.activation_index, 'direction': direction,
                    'name': self.index_tracker}
            size, dtype, count, memory = make_stats(tensor)
            stats = {'size': size, 'dtype': dtype, 'count': count, 'memory': memory}
            if offset and name in summary:
                info['batch_factor'] = [size[i] / summary[name]['stats']['size'][i] for i in range(len(size))]
            else:
                info['batch_factor'] = None
            summary[name] = {'info': info, 'stats': stats}
            self.activation_index += 1
            return tensor

        super().__init__(pack, unpack)


def make_stats(input):
    size = list(input.size())
    dtype = input.dtype
    count = input.numel()
    element_size = input.element_size()
    memory = count * element_size
    return size, dtype, count, memory
