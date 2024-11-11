import logging
import torch
from .cpu import CPUReport
from .cuda import CUDAReport


class Monitor:
    def __init__(self, cfg):
        self.cfg = cfg

    def hardware(self):
        status = {}
        status['cpu'] = CPUReport().get_report()
        if self.cfg['hardware']['cuda']['is_available']:
            status['cuda'] = CUDAReport().get_report()
        return status


def report_hardware(log=False):
    status = {}
    status['cpu'] = CPUReport()
    msg = f'CPU Info: {status['cpu']}'
    if log:
        logger = logging.getLogger('main')
        logger.info(msg)
    else:
        print(msg)
    if torch.cuda.is_available():
        status['cuda'] = CUDAReport()
        msg = f'CUDA Info: {status['cuda']}'
        if log:
            logger = logging.getLogger('main')
            logger.info(msg)
        else:
            print(msg)
    return


def update_hardware(status):
    status['cpu'] = CPUReport().get_report()
    if 'cuda' in status:
        status['cuda'] = CUDAReport().get_report()
    return status
