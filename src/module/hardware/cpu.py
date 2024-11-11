import psutil
import time


class CPUInfo:
    def __init__(self, name, total_memory, free_memory, used_memory, memory_utilization, cpu_utilization,
                 cores, temperature=None):
        self.name = name
        self.total_memory = total_memory
        self.free_memory = free_memory
        self.used_memory = used_memory
        self.memory_utilization = memory_utilization
        self.cpu_utilization = cpu_utilization
        self.cores = cores
        self.temperature = temperature

    def state_dict(self):
        return {
            'name': self.name,
            'total_memory': self.total_memory,
            'free_memory': self.free_memory,
            'used_memory': self.used_memory,
            'memory_utilization': self.memory_utilization,
            'cpu_utilization': self.cpu_utilization,
            'cores': self.cores,
            'temperature': self.temperature
        }

    def __repr__(self):
        temp_info = f', Temperature: {self.temperature:.2f} C' if self.temperature is not None else ''
        return (f'{self.name}: '
                f'Total Memory: {self.total_memory:.2f} MB, '
                f'Free Memory: {self.free_memory:.2f} MB, '
                f'Used Memory: {self.used_memory:.2f} MB, '
                f'Memory Utilization: {self.memory_utilization:.2f}%, '
                f'CPU Utilization: {self.cpu_utilization:.2f}%, '
                f'Cores: {self.cores}'
                f'{temp_info}')


class CPUReport:
    def __init__(self, interval=0.1, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def make_info(self):
        total_memory_samples = []
        free_memory_samples = []
        used_memory_samples = []
        memory_util_samples = []
        cpu_util_samples = []
        temperature_samples = []

        try:
            for _ in range(self.num_samples):
                virtual_memory = psutil.virtual_memory()
                total_memory_samples.append(virtual_memory.total / 1024 ** 2)
                free_memory_samples.append(virtual_memory.available / 1024 ** 2)
                used_memory_samples.append(virtual_memory.used / 1024 ** 2)
                memory_util_samples.append(virtual_memory.percent)
                cpu_util_samples.append(psutil.cpu_percent(interval=None))

                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures().get('coretemp', [])
                    if temps:
                        temperature_samples.append(sum(temp.current for temp in temps) / len(temps))
                else:
                    temperature_samples.append(None)

                time.sleep(self.interval)

            temperature_avg = (sum(t for t in temperature_samples if t is not None) /
                               len([t for t in temperature_samples if t is not None])
                               if any(t is not None for t in temperature_samples) else None)

            cores = psutil.cpu_count(logical=True)

            info = CPUInfo(
                name='CPU',
                total_memory=sum(total_memory_samples) / self.num_samples,
                free_memory=sum(free_memory_samples) / self.num_samples,
                used_memory=sum(used_memory_samples) / self.num_samples,
                memory_utilization=sum(memory_util_samples) / self.num_samples,
                cpu_utilization=sum(cpu_util_samples) / self.num_samples,
                cores=cores,
                temperature=temperature_avg
            )
        except Exception as e:
            raise ValueError(f'Failed to retrieve CPU information: {str(e)}')

        return info

    def get_report(self):
        return self.info.state_dict()

    def __repr__(self):
        return repr(self.info)
