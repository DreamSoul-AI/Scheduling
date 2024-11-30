import logging
import pynvml
import torch
import time
import numpy as np
from torch.cuda import cudart


class CUDAInfo:
    def __init__(self, static, monitor):
        self.static = static
        self.monitor = monitor

    def state_dict(self):
        return {
            'static': self.static,
            'monitor': self.monitor,
        }

    def __repr__(self):
        static_repr = ", ".join(f"{key}: {value}" for key, value in self.static.items())
        monitor_repr = ", ".join(f"{key}: {value}" for key, value in self.monitor.items())
        return f"Static Info: {static_repr}\nMonitoring Info: {monitor_repr}"


class CUDAReport:
    def __init__(self, interval=0.1, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def compute_mean(self, data):
        if len(data) > 0:
            return np.mean(data, axis=0).tolist()
        return None

    def make_static_info(self, handle, device):
        """Retrieve static info for a specific GPU."""
        static_info = {}

        # Get basic information about the GPU
        static_info['name'] = pynvml.nvmlDeviceGetName(handle)
        static_info['driver_version'] = pynvml.nvmlSystemGetDriverVersion()
        static_info['pci_bus_id'] = pynvml.nvmlDeviceGetPciInfo(handle).busId

        # Get the architecture's compute capability (major, minor)
        device_properties = torch.cuda.get_device_properties(device)
        static_info['architecture_major'] = device_properties.major
        static_info['architecture_minor'] = device_properties.minor

        # Get the number of SMs (Streaming Multiprocessors)
        sm_count = device_properties.multi_processor_count
        static_info['sm_count'] = sm_count

        # Get the max clock rate (SM clock)
        sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        static_info['sm_clock'] = sm_clock  # in MHz
        return static_info

    def make_monitor_info(self, handle):
        """Retrieve monitoring info (dynamic stats) for a specific GPU."""
        memory_info = None
        memory_samples = []
        utilization_samples = []
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
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # in W
            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

            # NVLink Throughput (if available)
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
            memory_samples.append(memory_info.used / 1024 ** 2)  # in MB
            utilization_samples.append(utilization.gpu)
            temperature_samples.append(temperature)
            power_usage_samples.append(power_usage)
            pcie_rx_samples.append(pcie_rx)
            pcie_tx_samples.append(pcie_tx)
            graphics_clock_samples.append(graphics_clock)
            memory_clock_samples.append(memory_clock)
            nvlink_rx_samples.append(nvlink_rx / 1024)  # in KB/s
            nvlink_tx_samples.append(nvlink_tx / 1024)  # in KB/s

            time.sleep(self.interval)

        monitor_info = {
            "memory_total": memory_info.total / 1024 ** 2,  # in MB
            "memory_free": memory_info.free / 1024 ** 2,
            "memory_used": memory_info.used / 1024 ** 2,
            "memory_utilization": self.compute_mean(memory_samples),
            "gpu_utilization": self.compute_mean(utilization_samples),
            "temperature": self.compute_mean(temperature_samples),
            "power_usage": self.compute_mean(power_usage_samples),
            "pcie_rx_rate": self.compute_mean(pcie_rx_samples),
            "pcie_tx_rate": self.compute_mean(pcie_tx_samples),
            "graphics_clock": self.compute_mean(graphics_clock_samples),
            "memory_clock": self.compute_mean(memory_clock_samples),
            "nvlink_rx": self.compute_mean(nvlink_rx_samples),
            "nvlink_tx": self.compute_mean(nvlink_tx_samples),
        }
        return monitor_info

    def make_info(self):
        """Fetch both static and monitoring info for all GPUs."""
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
                device = torch.cuda.device(i)

                # Get static and monitoring information
                static_info = self.make_static_info(handle, device)
                monitor_info = self.make_monitor_info(handle)

                # Create CUDAInfo object combining static and monitoring info
                info_i = CUDAInfo(static=static_info, monitor=monitor_info)
                info.append(info_i)
        except pynvml.NVMLError as e:
            raise ValueError(f'Failed to retrieve GPU information: {str(e)}')
        finally:
            pynvml.nvmlShutdown()

        return info

    def get_report(self):
        return [info.state_dict() for info in self.info]

    def __repr__(self):
        return '\n'.join(repr(info) for info in self.info)
