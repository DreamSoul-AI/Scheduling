import logging
import pynvml
import time
import numpy as np


class CUDAInfo:
    def __init__(self, index, name, total_memory, free_memory, used_memory, memory_utilization, gpu_utilization,
                 temperature, power_usage, pcie_rx_rate, pcie_tx_rate, graphics_clock, memory_clock, nvlink_rx,
                 nvlink_tx):
        self.index = index
        self.name = name
        self.total_memory = total_memory
        self.free_memory = free_memory
        self.used_memory = used_memory
        self.memory_utilization = memory_utilization
        self.gpu_utilization = gpu_utilization
        self.temperature = temperature
        self.power_usage = power_usage
        self.pcie_rx_rate = pcie_rx_rate
        self.pcie_tx_rate = pcie_tx_rate
        self.graphics_clock = graphics_clock
        self.memory_clock = memory_clock
        self.nvlink_rx = nvlink_rx
        self.nvlink_tx = nvlink_tx

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
            'power_usage': self.power_usage,
            'pcie_rx_rate': self.pcie_rx_rate,
            'pcie_tx_rate': self.pcie_tx_rate,
            'graphics_clock': self.graphics_clock,
            'memory_clock': self.memory_clock,
            'nvlink_rx': self.nvlink_rx,
            'nvlink_tx': self.nvlink_tx,
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
            f'Power Usage: {self.power_usage:.2f} W, '
            f'PCIe RX Rate: {self.pcie_rx_rate:.2f} KB/s, '
            f'PCIe TX Rate: {self.pcie_tx_rate:.2f} KB/s, '
            f'Graphics Clock: {self.graphics_clock} MHz, '
            f'Memory Clock: {self.memory_clock} MHz, '
            f'NVLink RX: {self.nvlink_rx} KB/s, NVLink TX: {self.nvlink_tx} KB/s'
        )


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
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)

                total_memory_samples = []
                free_memory_samples = []
                used_memory_samples = []
                memory_util_samples = []
                gpu_util_samples = []
                temperature_samples = []
                power_usage_samples = []
                pcie_rx_samples = []
                pcie_tx_samples = []
                graphics_clock_samples = []
                memory_clock_samples = []
                nvlink_rx_samples = []
                nvlink_tx_samples = []

                for _ in range(self.num_samples):
                    # NVML Metrics
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                    pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                    pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

                    # NVLink Throughput
                    nvlink_rx, nvlink_tx = 0, 0
                    try:
                        for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                            if pynvml.nvmlDeviceGetNvLinkState(handle, link):
                                nvlink_rx += pynvml.nvmlDeviceGetNvLinkThroughput(handle, link,
                                                                                  pynvml.NVML_NVLINK_THROUGHPUT_RX)
                                nvlink_tx += pynvml.nvmlDeviceGetNvLinkThroughput(handle, link,
                                                                                  pynvml.NVML_NVLINK_THROUGHPUT_TX)
                    except pynvml.NVMLError:
                        pass

                    # Append to samples
                    total_memory_samples.append(memory_info.total / 1024 ** 2)
                    free_memory_samples.append(memory_info.free / 1024 ** 2)
                    used_memory_samples.append(memory_info.used / 1024 ** 2)
                    memory_util_samples.append(memory_info.used / memory_info.total * 100)
                    gpu_util_samples.append(utilization.gpu)
                    temperature_samples.append(temperature)
                    power_usage_samples.append(power_usage)
                    pcie_rx_samples.append(pcie_rx)
                    pcie_tx_samples.append(pcie_tx)
                    graphics_clock_samples.append(graphics_clock)
                    memory_clock_samples.append(memory_clock)
                    nvlink_rx_samples.append(nvlink_rx / 1024)  # Convert to KB/s
                    nvlink_tx_samples.append(nvlink_tx / 1024)  # Convert to KB/s

                    time.sleep(self.interval)

                # Create CUDAInfo object
                info_i = CUDAInfo(
                    index=i,
                    name=name,
                    total_memory=self.compute_mean(total_memory_samples),
                    free_memory=self.compute_mean(free_memory_samples),
                    used_memory=self.compute_mean(used_memory_samples),
                    memory_utilization=self.compute_mean(memory_util_samples),
                    gpu_utilization=self.compute_mean(gpu_util_samples),
                    temperature=self.compute_mean(temperature_samples),
                    power_usage=self.compute_mean(power_usage_samples),
                    pcie_rx_rate=self.compute_mean(pcie_rx_samples),
                    pcie_tx_rate=self.compute_mean(pcie_tx_samples),
                    graphics_clock=self.compute_mean(graphics_clock_samples),
                    memory_clock=self.compute_mean(memory_clock_samples),
                    nvlink_rx=self.compute_mean(nvlink_rx_samples),
                    nvlink_tx=self.compute_mean(nvlink_tx_samples),
                )
                info.append(info_i)
        except pynvml.NVMLError as e:
            raise ValueError(f'Failed to retrieve GPU information: {str(e)}')
        finally:
            pynvml.nvmlShutdown()

        return info

    def compute_mean(self, data):
        if len(data) > 0:
            return np.mean(data, axis=0).tolist()
        return None

    def get_report(self):
        return [info.state_dict() for info in self.info]

    def __repr__(self):
        return '\n'.join(repr(info) for info in self.info)
