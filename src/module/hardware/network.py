import psutil
import socket
import time
import numpy as np


class NetworkInfo:
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


class NetworkReport:
    def __init__(self, interval=1.0, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def compute_mean(self, data):
        return np.mean(data, axis=0).tolist() if data else None

    def calculate_rate(self, samples):
        """
        Calculate the average rate of change for the given samples.
        :param samples: List of cumulative values (e.g., bytes sent/received).
        :return: Average rate of change (per second).
        """
        if len(samples) < 2:
            return 0
        samples = np.array(samples)
        rates = np.diff(samples) / self.interval  # Calculate differences and normalize by interval
        return np.mean(rates).item()  # Return the average rate

    def make_static_info(self):
        network_interfaces = psutil.net_if_addrs()
        network_stats = psutil.net_if_stats()

        static_info = {}

        # Gather detailed information about each interface
        for interface in network_interfaces:
            # IP, MAC, IPv6 for each interface
            ip_info = {"IPv4": None, "IPv6": None, "MAC": None}
            for addr in network_interfaces[interface]:
                # Handle valid address families: AF_INET, AF_INET6, and AF_LINK (MAC address)
                if addr.family == socket.AF_INET:
                    ip_info["IPv4"] = addr.address
                elif addr.family == socket.AF_INET6:
                    ip_info["IPv6"] = addr.address
                elif addr.family == socket.AF_LINK or addr.family == -1:
                    # Handle AF_LINK (MAC address)
                    ip_info["MAC"] = addr.address.replace('-', ':')

            # Interface stats (up/down, speed, duplex)
            stats = network_stats.get(interface, None)
            if stats:
                interface_status = "up" if stats.isup else "down"
                interface_speed = stats.speed  # in Mbps
                interface_duplex = stats.duplex  # Full or Half duplex
            else:
                interface_status = "unknown"
                interface_speed = "unknown"
                interface_duplex = "unknown"

            # Use the actual interface name as the key
            static_info[interface] = {
                "ip_info": ip_info,
                "status": interface_status,
                "speed": interface_speed,
                "duplex": interface_duplex
            }
        return static_info

    def make_monitor_info(self):
        sent_samples = []
        recv_samples = []
        packets_sent_samples = []
        packets_recv_samples = []
        err_in_samples = []
        err_out_samples = []
        drop_in_samples = []
        drop_out_samples = []

        for _ in range(self.num_samples):
            net_stats = psutil.net_io_counters()
            sent_samples.append(net_stats.bytes_sent / 1024 ** 2)  # Convert to MB
            recv_samples.append(net_stats.bytes_recv / 1024 ** 2)  # Convert to MB
            packets_sent_samples.append(net_stats.packets_sent)
            packets_recv_samples.append(net_stats.packets_recv)
            err_in_samples.append(net_stats.errin)
            err_out_samples.append(net_stats.errout)
            drop_in_samples.append(net_stats.dropin)
            drop_out_samples.append(net_stats.dropout)

            time.sleep(self.interval)

        # Calculate rates for each sample
        sent_rate = self.calculate_rate(sent_samples)
        recv_rate = self.calculate_rate(recv_samples)
        packets_sent_rate = self.calculate_rate(packets_sent_samples)
        packets_recv_rate = self.calculate_rate(packets_recv_samples)
        err_in_rate = self.calculate_rate(err_in_samples)
        err_out_rate = self.calculate_rate(err_out_samples)
        drop_in_rate = self.calculate_rate(drop_in_samples)
        drop_out_rate = self.calculate_rate(drop_out_samples)

        monitor_info = {
            "sent_rate": sent_rate,
            "recv_rate": recv_rate,
            "packets_sent_rate": packets_sent_rate,
            "packets_recv_rate": packets_recv_rate,
            "err_in_rate": err_in_rate,
            "err_out_rate": err_out_rate,
            "drop_in_rate": drop_in_rate,
            "drop_out_rate": drop_out_rate,
        }
        return monitor_info

    def make_info(self):
        static_info = self.make_static_info()
        monitor_info = self.make_monitor_info()
        return NetworkInfo(static=static_info, monitor=monitor_info)

    def get_report(self):
        return self.info.state_dict()

    def __repr__(self):
        return repr(self.info)
