import psutil
import time
import numpy as np


class NetworkInfo:
    def __init__(self, name, sent_rate, recv_rate, packets_sent_rate, packets_recv_rate,
                 err_in_rate, err_out_rate, drop_in_rate, drop_out_rate):
        self.name = name
        self.sent_rate = sent_rate
        self.recv_rate = recv_rate
        self.packets_sent_rate = packets_sent_rate
        self.packets_recv_rate = packets_recv_rate
        self.err_in_rate = err_in_rate
        self.err_out_rate = err_out_rate
        self.drop_in_rate = drop_in_rate
        self.drop_out_rate = drop_out_rate

    def state_dict(self):
        return {
            'name': self.name,
            'sent_rate': self.sent_rate,
            'recv_rate': self.recv_rate,
            'packets_sent_rate': self.packets_sent_rate,
            'packets_recv_rate': self.packets_recv_rate,
            'err_in_rate': self.err_in_rate,
            'err_out_rate': self.err_out_rate,
            'drop_in_rate': self.drop_in_rate,
            'drop_out_rate': self.drop_out_rate,
        }

    def __repr__(self):
        return (f"{self.name}: "
                f"Send Rate: {self.sent_rate:.2f} MB/s, "
                f"Receive Rate: {self.recv_rate:.2f} MB/s, "
                f"Packets Sent Rate: {self.packets_sent_rate:.2f}/s, "
                f"Packets Received Rate: {self.packets_recv_rate:.2f}/s, "
                f"Error In Rate: {self.err_in_rate:.2f}/s, "
                f"Error Out Rate: {self.err_out_rate:.2f}/s, "
                f"Dropped In Rate: {self.drop_in_rate:.2f}/s, "
                f"Dropped Out Rate: {self.drop_out_rate:.2f}/s")


class NetworkReport:
    def __init__(self, interval=1.0, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def make_info(self):
        try:
            # Initialize cumulative sample containers
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

            # Compute rates using differences
            sent_rate = self.calculate_rate(sent_samples)
            recv_rate = self.calculate_rate(recv_samples)
            packets_sent_rate = self.calculate_rate(packets_sent_samples)
            packets_recv_rate = self.calculate_rate(packets_recv_samples)
            err_in_rate = self.calculate_rate(err_in_samples)
            err_out_rate = self.calculate_rate(err_out_samples)
            drop_in_rate = self.calculate_rate(drop_in_samples)
            drop_out_rate = self.calculate_rate(drop_out_samples)

            return NetworkInfo(
                name="Network",
                sent_rate=sent_rate,
                recv_rate=recv_rate,
                packets_sent_rate=packets_sent_rate,
                packets_recv_rate=packets_recv_rate,
                err_in_rate=err_in_rate,
                err_out_rate=err_out_rate,
                drop_in_rate=drop_in_rate,
                drop_out_rate=drop_out_rate,
            )

        except Exception as e:
            raise ValueError(f"Failed to retrieve network information: {str(e)}")

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

    def get_report(self):
        return self.info.state_dict()

    def __repr__(self):
        return repr(self.info)
