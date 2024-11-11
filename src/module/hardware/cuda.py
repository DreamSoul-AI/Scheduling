import logging
import pynvml
import time


class CUDAInfo:
    def __init__(self, index, name, total_memory, free_memory, used_memory, memory_utilization, gpu_utilization,
                 temperature, power_usage):
        self.index = index
        self.name = name
        self.total_memory = total_memory
        self.free_memory = free_memory
        self.used_memory = used_memory
        self.memory_utilization = memory_utilization
        self.gpu_utilization = gpu_utilization
        self.temperature = temperature
        self.power_usage = power_usage

    def state_dict(self):
        return {
            'index': self.index,
            'name': self.name,
            'total_memory': self.total_memory,
            'free_memory': self.free_memory,
            'used_memory': self.used_memory,
            'memory_utilization': self.memory_utilization,
            'gpu_utilization': self.gpu_utilization,
            'temperature': self.temperature,
            'power_usage': self.power_usage
        }

    def __repr__(self):
        return (
            f'GPU {self.index} ({self.name}): '
            f'Total Memory: {self.total_memory:.2f} MB, '
            f'Free Memory: {self.free_memory:.2f} MB, '
            f'Used Memory: {self.used_memory:.2f} MB, '
            f'Memory Utilization: {self.memory_utilization:.2f}%, '
            f'GPU Utilization: {self.gpu_utilization:.2f}%, '
            f'Temperature: {self.temperature:.2f} C, '
            f'Power Usage: {self.power_usage:.2f} W')


class CUDAReport:
    def __init__(self, interval=0.1, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def make_info(self):
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            logger = logging.getLogger('main')
            logger.error(f'Failed to initialize NVML: {str(e)}')
            return []

        info = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                total_memory_samples = []
                free_memory_samples = []
                used_memory_samples = []
                memory_util_samples = []
                gpu_util_samples = []
                temperature_samples = []
                power_usage_samples = []

                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)

                for _ in range(self.num_samples):
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000

                    total_memory_samples.append(memory_info.total / 1024 ** 2)
                    free_memory_samples.append(memory_info.free / 1024 ** 2)
                    used_memory_samples.append(memory_info.used / 1024 ** 2)
                    memory_util_samples.append(memory_info.used / memory_info.total * 100)
                    gpu_util_samples.append(utilization.gpu * 100)
                    temperature_samples.append(temperature)
                    power_usage_samples.append(power_usage)

                    time.sleep(self.interval)

                info_i = CUDAInfo(
                    index=i,
                    name=name,
                    total_memory=sum(total_memory_samples) / self.num_samples,
                    free_memory=sum(free_memory_samples) / self.num_samples,
                    used_memory=sum(used_memory_samples) / self.num_samples,
                    memory_utilization=sum(memory_util_samples) / self.num_samples,
                    gpu_utilization=sum(gpu_util_samples) / self.num_samples,
                    temperature=sum(temperature_samples) / self.num_samples,
                    power_usage=sum(power_usage_samples) / self.num_samples
                )
                info.append(info_i)
        except pynvml.NVMLError as e:
            raise ValueError('Failed to retrieve GPU information: {}'.format(str(e)))
        finally:
            pynvml.nvmlShutdown()

        return info

    def get_report(self):
        return [info.state_dict() for info in self.info]

    def __repr__(self):
        return '\n'.join(repr(info) for info in self.info)
