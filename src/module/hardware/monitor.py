import logging
import torch
from .cpu import CPUReport
from .cuda import CUDAReport
from .disk import DiskReport
from .network import NetworkReport


class Monitor:
    def hardware(self):
        status = {}
        status['cpu'] = CPUReport().get_report()
        if torch.cuda.is_available:
            status['cuda'] = CUDAReport().get_report()
        status['disk'] = DiskReport().get_report()
        status['network'] = NetworkReport().get_report()
        print(status['cpu'])
        exit()
        return status


def report_hardware(log=False):
    def make_msg(msg):
        if log:
            logger = logging.getLogger('main')
            logger.info(msg)
        else:
            print(msg)
        return

    status = {}
    status['cpu'] = CPUReport()
    make_msg(f'CPU Info: {status['cpu']}')
    if torch.cuda.is_available():
        status['cuda'] = CUDAReport()
        make_msg(f'CUDA Info: {status['cuda']}')
    status['disk'] = DiskReport()
    make_msg(f'Disk Info: {status['disk']}')
    return


def update_hardware(status):
    status['cpu'] = CPUReport().get_report()
    if 'cuda' in status:
        status['cuda'] = CUDAReport().get_report()
    return status
