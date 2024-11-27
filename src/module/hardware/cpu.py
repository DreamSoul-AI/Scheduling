import psutil
import time
import numpy as np
import cpuinfo


class CPUInfo:
    def __init__(self, static, monitor):
        self.static = static
        self.monitor = monitor

    def state_dict(self):
        return {
            "static": self.static,
            "monitor": self.monitor,
        }

    def __repr__(self):
        static_repr = ", ".join(f"{key}: {value}" for key, value in self.static.items())
        monitor_repr = ", ".join(f"{key}: {value}" for key, value in self.monitor.items())
        return f"Static Info: {static_repr}\nMonitoring Info: {monitor_repr}"


class CPUReport:
    def __init__(self, interval=0.1, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def compute_mean(self, data):
        return np.mean(data, axis=0).tolist() if data else None

    def make_static_info(self):
        cores = psutil.cpu_count(logical=True)
        cpu_info = cpuinfo.get_cpu_info()
        static_info = {"name": "CPU", "cores": cores, **cpu_info}
        return static_info

    def make_monitor_info(self):
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

        for _ in range(self.num_samples):
            virtual_memory = psutil.virtual_memory()
            total_memory_samples.append(virtual_memory.total / 1024 ** 2)
            free_memory_samples.append(virtual_memory.available / 1024 ** 2)
            used_memory_samples.append(virtual_memory.used / 1024 ** 2)
            memory_util_samples.append(virtual_memory.percent)

            overall_cpu_util_samples.append(psutil.cpu_percent(interval=None))
            per_cpu_util_samples.append(psutil.cpu_percent(interval=None, percpu=True))

            overall_cpu_freq_samples.append(psutil.cpu_freq().current)
            per_cpu_freq_samples.append([freq.current for freq in psutil.cpu_freq(percpu=True)])

            stats = psutil.cpu_stats()
            ctx_switches_samples.append(stats.ctx_switches)
            interrupts_samples.append(stats.interrupts)

            num_threads_samples.append(psutil.cpu_count(logical=True))
            num_processes_samples.append(len(psutil.pids()))

            paging_stats = psutil.swap_memory()
            page_ins_samples.append(paging_stats.sin)
            page_outs_samples.append(paging_stats.sout)

            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures().get('coretemp', [])
                if temps:
                    temperature_samples.append(sum(temp.current for temp in temps) / len(temps))
                else:
                    temperature_samples.append(None)

            time.sleep(self.interval)
        monitor_info = {
            "total_memory": self.compute_mean(total_memory_samples),
            "free_memory": self.compute_mean(free_memory_samples),
            "used_memory": self.compute_mean(used_memory_samples),
            "memory_utilization": self.compute_mean(memory_util_samples),
            "cpu_utilization": self.compute_mean(overall_cpu_util_samples),
            "per_cpu_utilization": self.compute_mean(per_cpu_util_samples),
            "cpu_frequencies": self.compute_mean(overall_cpu_freq_samples),
            "per_cpu_frequencies": self.compute_mean(per_cpu_freq_samples),
            "temperature": self.compute_mean([t for t in temperature_samples if t is not None]),
            "ctx_switches": self.compute_mean(ctx_switches_samples),
            "interrupts": self.compute_mean(interrupts_samples),
            "num_threads": self.compute_mean(num_threads_samples),
            "num_processes": self.compute_mean(num_processes_samples),
            "page_ins": self.compute_mean(page_ins_samples),
            "page_outs": self.compute_mean(page_outs_samples),
        }
        return monitor_info

    def make_info(self):
        static_info = self.make_static_info()
        monitor_info = self.make_monitor_info()
        return CPUInfo(static=static_info, monitor=monitor_info)

    def get_report(self):
        return self.info.state_dict()

    def __repr__(self):
        return repr(self.info)
