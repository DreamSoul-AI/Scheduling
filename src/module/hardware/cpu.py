import psutil
import time
import numpy as np
import cpuinfo


class CPUInfo:
    def __init__(self, name, cores, total_memory, free_memory, used_memory, memory_utilization,
                 cpu_utilization, per_cpu_utilization, cpu_frequencies, per_cpu_frequencies,
                 temperature, ctx_switches, interrupts, num_threads, num_processes,
                 page_ins, page_outs, static_info):
        self.name = name
        self.cores = cores
        self.total_memory = total_memory
        self.free_memory = free_memory
        self.used_memory = used_memory
        self.memory_utilization = memory_utilization
        self.cpu_utilization = cpu_utilization
        self.per_cpu_utilization = per_cpu_utilization
        self.cpu_frequencies = cpu_frequencies
        self.per_cpu_frequencies = per_cpu_frequencies
        self.temperature = temperature
        self.ctx_switches = ctx_switches
        self.interrupts = interrupts
        self.num_threads = num_threads
        self.num_processes = num_processes
        self.page_ins = page_ins
        self.page_outs = page_outs
        self.static_info = static_info

    def state_dict(self):
        return {
            'name': self.name,
            'cores': self.cores,
            'total_memory': self.total_memory,
            'free_memory': self.free_memory,
            'used_memory': self.used_memory,
            'memory_utilization': self.memory_utilization,
            'cpu_utilization': self.cpu_utilization,
            'per_cpu_utilization': self.per_cpu_utilization,
            'cpu_frequencies': self.cpu_frequencies,
            'per_cpu_frequencies': self.per_cpu_frequencies,
            'temperature': self.temperature,
            'ctx_switches': self.ctx_switches,
            'interrupts': self.interrupts,
            'num_threads': self.num_threads,
            'num_processes': self.num_processes,
            'page_ins': self.page_ins,
            'page_outs': self.page_outs,
            'static_info': self.static_info
        }

    def __repr__(self):
        repr_str = (f'{self.name}: '
                    f'Cores: {self.cores}, '
                    f'Total Memory: {self.total_memory:.2f} MB, '
                    f'Free Memory: {self.free_memory:.2f} MB, '
                    f'Used Memory: {self.used_memory:.2f} MB, '
                    f'Memory Utilization: {self.memory_utilization:.2f}%, '
                    f'CPU Utilization: {self.cpu_utilization:.2f}%, '
                    f'Per-Core Utilization: {self.per_cpu_utilization}, '
                    f'CPU Frequencies: {self.cpu_frequencies:.2f} MHz, '
                    f'Per-Core Frequencies: {self.per_cpu_frequencies}, '
                    f'Temperature: {self.temperature:.2f} C, '
                    f'Context Switches: {self.ctx_switches}, '
                    f'Interrupts: {self.interrupts}, '
                    f'Num Threads: {self.num_threads}, '
                    f'Num Processes: {self.num_processes}, '
                    f'Page Ins: {self.page_ins}, '
                    f'Page Outs: {self.page_outs}, ')

        if self.static_info is not None:
            static_info_str = ", ".join([f'{key}: {value}' for key, value in self.static_info.items()])
            repr_str += f'Static Info: {static_info_str}'

        return repr_str


class CPUReport:
    def __init__(self, interval=0.1, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def compute_mean(self, data):
        if len(data) > 0:
            return np.mean(data, axis=0).tolist()
        return None

    def make_info(self):
        total_memory_samples = []
        free_memory_samples = []
        used_memory_samples = []
        memory_util_samples = []
        overall_cpu_util_samples = []
        per_cpu_util_samples = []
        overall_cpu_freq_samples = []
        per_cpu_freq_samples = []
        temperature_samples = []
        ctx_switches_samples = []
        interrupts_samples = []
        num_threads_samples = []
        num_processes_samples = []
        page_ins_samples = []
        page_outs_samples = []

        try:
            for _ in range(self.num_samples):
                virtual_memory = psutil.virtual_memory()
                total_memory_samples.append(virtual_memory.total / 1024 ** 2)
                free_memory_samples.append(virtual_memory.available / 1024 ** 2)
                used_memory_samples.append(virtual_memory.used / 1024 ** 2)
                memory_util_samples.append(virtual_memory.percent)

                # CPU utilization
                overall_cpu_util_samples.append(psutil.cpu_percent(interval=None))
                per_cpu_util_samples.append(psutil.cpu_percent(interval=None, percpu=True))

                # CPU frequencies
                overall_cpu_freq_samples.append(psutil.cpu_freq().current)
                per_cpu_freq_samples.append([freq.current for freq in psutil.cpu_freq(percpu=True)])

                stats = psutil.cpu_stats()
                ctx_switches_samples.append(stats.ctx_switches)
                interrupts_samples.append(stats.interrupts)

                # Thread/process monitoring
                num_threads_samples.append(psutil.cpu_count(logical=True))
                num_processes_samples.append(len(psutil.pids()))

                # Paging (swap) statistics
                paging_stats = psutil.swap_memory()
                page_ins_samples.append(paging_stats.sin)
                page_outs_samples.append(paging_stats.sout)

                # Temperature readings
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures().get('coretemp', [])
                    if temps:
                        temperature_samples.append(sum(temp.current for temp in temps) / len(temps))
                else:
                    temperature_samples.append(None)

                time.sleep(self.interval)

            cores = psutil.cpu_count(logical=True)

            # Get static CPU information
            static_info = cpuinfo.get_cpu_info()

            # Process all samples to compute means
            info = CPUInfo(
                name='CPU',
                cores=cores,
                total_memory=self.compute_mean(total_memory_samples),
                free_memory=self.compute_mean(free_memory_samples),
                used_memory=self.compute_mean(used_memory_samples),
                memory_utilization=self.compute_mean(memory_util_samples),
                cpu_utilization=self.compute_mean(overall_cpu_util_samples),
                per_cpu_utilization=self.compute_mean(per_cpu_util_samples),
                cpu_frequencies=self.compute_mean(overall_cpu_freq_samples),
                per_cpu_frequencies=self.compute_mean(per_cpu_freq_samples),
                temperature=self.compute_mean([t for t in temperature_samples if t is not None]),
                ctx_switches=self.compute_mean(ctx_switches_samples),
                interrupts=self.compute_mean(interrupts_samples),
                num_threads=self.compute_mean(num_threads_samples),
                num_processes=self.compute_mean(num_processes_samples),
                page_ins=self.compute_mean(page_ins_samples),
                page_outs=self.compute_mean(page_outs_samples),
                static_info=static_info
            )
            return info

        except Exception as e:
            raise ValueError(f'Failed to retrieve CPU information: {str(e)}')

    def get_report(self):
        return self.info.state_dict()

    def __repr__(self):
        return repr(self.info)
