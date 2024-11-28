import os
import platform
import psutil
import time
import numpy as np

# Import WMI for Windows
try:
    import wmi
except ImportError:
    wmi = None


class DiskInfo:
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


class DiskReport:
    def __init__(self, interval=0.1, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def compute_mean(self, data):
        return np.mean(data, axis=0).tolist() if data else None

    def make_static_info(self):
        if platform.system() == "Windows":
            return self.make_static_info_windows()  # Call the Windows-specific method
        elif platform.system() == "Linux":
            return self.make_static_info_linux()  # Call the Linux-specific method
        else:
            raise NotImplementedError(f"Unsupported platform: {platform.system()}")

    def make_static_info_windows(self):
        """
        Get static disk info on Windows using WMI.
        """
        if wmi is None:
            raise ImportError("WMI is required for this functionality. Please install the `wmi` module.")

        disk_info_list = []
        c = wmi.WMI()

        for disk in c.Win32_DiskDrive():
            # Directly use the WMI attributes without parsing or additional processing
            static_info = {
                "DeviceID": disk.DeviceID,
                "Model": disk.Model,
                "Manufacturer": disk.Manufacturer,
                "SerialNumber": disk.SerialNumber.strip(),  # Ensure no extra spaces
                "Size": disk.Size,  # Size in bytes
                "Partitions": disk.Partitions,
                "MediaType": disk.MediaType,
                "FirmwareRevision": disk.FirmwareRevision,
                "InterfaceType": disk.InterfaceType,
                "Caption": disk.Caption,  # Caption provides a user-friendly name
                "Status": disk.Status,
            }

            disk_info_list.append(static_info)

        return disk_info_list

    def make_static_info_linux(self):
        """
        Get static disk info on Linux using psutil and /sys.
        """
        disk_info_list = []
        partitions = psutil.disk_partitions()

        for partition in partitions:
            device_name = partition.device

            # Derive the physical device (e.g., /dev/sda1 -> /dev/sda)
            if device_name.startswith("/dev/"):
                physical_device = device_name.rstrip("1234567890")
                if not os.path.exists(f"/sys/block/{physical_device}"):
                    continue

                # Read the additional info from /sys/block/<device>
                try:
                    # Get the manufacturer, model, and serial number
                    with open(f"/sys/block/{physical_device}/device/vendor", "r") as f:
                        manufacturer = f.read().strip()
                    with open(f"/sys/block/{physical_device}/device/model", "r") as f:
                        model = f.read().strip()
                    with open(f"/sys/block/{physical_device}/device/serial", "r") as f:
                        serial_number = f.read().strip()

                    # Create static info for this disk
                    static_info = {
                        "name": physical_device,
                        "model": model,
                        "manufacturer": manufacturer,
                        "serial_number": serial_number,
                        "partitions": [],
                    }
                    disk_info_list.append(static_info)

                except Exception as e:
                    print(f"Error processing disk {physical_device}: {e}")

        return disk_info_list

    def make_monitor_info(self):
        """
        Collects disk I/O and usage statistics, using device mappings for physical devices.
        """
        # Get the physical device to partition mapping
        physical_device_map = self._get_physical_device_map()

        # Initialize dictionaries to track read/write rates for physical devices
        disk_samples = {device.lower(): {'read_bytes': [], 'write_bytes': [],
                                         'read_count': [], 'write_count': [],
                                         'read_time': [], 'write_time': []}
                        for device in psutil.disk_io_counters(perdisk=True)}
        disk_info_list = []

        try:
            # Sampling loop
            for _ in range(self.num_samples):
                io_counters = psutil.disk_io_counters(perdisk=True)
                for device, stats in io_counters.items():
                    if device.lower() in disk_samples:
                        disk_samples[device.lower()]['read_bytes'].append(stats.read_bytes / 1024 ** 2)  # MB
                        disk_samples[device.lower()]['write_bytes'].append(stats.write_bytes / 1024 ** 2)  # MB
                        disk_samples[device.lower()]['read_count'].append(stats.read_count)
                        disk_samples[device.lower()]['write_count'].append(stats.write_count)
                        disk_samples[device.lower()]['read_time'].append(stats.read_time)
                        disk_samples[device.lower()]['write_time'].append(stats.write_time)

                time.sleep(self.interval)

            # Process each physical device
            for physical_device, partitions in physical_device_map.items():
                # Get samples for this device
                samples = disk_samples.get(physical_device, {'read_bytes': [], 'write_bytes': [],
                                                             'read_count': [], 'write_count': [],
                                                             'read_time': [], 'write_time': []})

                # Calculate read/write rates
                read_rate = self.calculate_rate(samples['read_bytes'])
                write_rate = self.calculate_rate(samples['write_bytes'])

                # Calculate average time per I/O operation
                read_latency = self.calculate_latency(samples['read_time'], samples['read_count'])
                write_latency = self.calculate_latency(samples['write_time'], samples['write_count'])

                # Calculate disk busy percentage
                read_busy_percentage = self.calculate_busy_percentage(samples['read_time'])
                write_busy_percentage = self.calculate_busy_percentage(samples['write_time'])

                # Combine space information from partitions using the aggregation function
                total_space, free_space, used_space, percent_used = self._aggregate_partition_usage(partitions)

                # Create the disk info object for this physical device
                monitor_info = {
                    "name": physical_device,
                    "total_space_GB": total_space,
                    "free_space_GB": free_space,
                    "used_space_GB": used_space,
                    "percent_used": percent_used,
                    "read_rate": read_rate,
                    "write_rate": write_rate,
                    "read_latency": read_latency,
                    "write_latency": write_latency,
                    "read_busy_percentage": read_busy_percentage,
                    "write_busy_percentage": write_busy_percentage,
                    "partitions": partitions
                }

                disk_info_list.append(monitor_info)

        except Exception as e:
            raise ValueError(f"Failed to retrieve Disk information: {str(e)}")

        return disk_info_list

    def make_info(self):
        static_info_list = self.make_static_info()
        monitor_info_list = self.make_monitor_info()
        disk_info_list = [
            DiskInfo(static=static_info, monitor=monitor_info)
            for static_info, monitor_info in zip(static_info_list, monitor_info_list)
        ]
        return disk_info_list

    def calculate_rate(self, samples):
        """
        Calculate the average rate of change for the given samples.
        :param samples: List of cumulative values (e.g., read/write bytes).
        :return: Average rate of change (per second).
        """
        if len(samples) < 2:
            return 0
        samples = np.array(samples)
        rates = np.diff(samples) / self.interval  # Calculate differences and normalize by interval
        return np.mean(rates).item()  # Return the average rate

    def calculate_latency(self, time_samples, count_samples):
        """
        Calculate the average latency for I/O operations using NumPy.
        :param time_samples: List of cumulative time spent on read/write operations (in milliseconds).
        :param count_samples: List of cumulative read/write operation counts.
        :return: Average latency per I/O operation (in milliseconds).
        """
        if len(time_samples) < 2 or len(count_samples) < 2:
            return 0
        time_samples = np.array(time_samples)
        count_samples = np.array(count_samples)

        # Calculate total time
        total_time = time_samples[-1] - time_samples[0]
        if total_time == 0:  # Fallback to len(samples) * interval if total time is zero
            total_time = len(time_samples) * self.interval * 1000  # Convert interval to milliseconds

        # Calculate total operations
        total_operations = count_samples[-1] - count_samples[0]
        if total_operations == 0:  # No operations, latency should be zero
            return 0

        return total_time / total_operations  # Average latency per operation

    def calculate_busy_percentage(self, time_samples):
        """
        Calculate the percentage of time the disk is busy handling a specific type of I/O (read or write).
        :param time_samples: List of cumulative time spent on read or write operations (in milliseconds).
        :return: Busy percentage over the sampling period.
        """
        if len(time_samples) < 2:
            return 0
        time_samples = np.array(time_samples)

        # Calculate total I/O time
        total_time = time_samples[-1] - time_samples[0]

        # Total sampling time in milliseconds
        total_sampling_time = self.interval * self.num_samples * 1000  # Convert seconds to milliseconds

        # Calculate busy percentage
        return (total_time / total_sampling_time) * 100 if total_sampling_time > 0 else 0

    def _get_physical_device_map(self):
        """
        Cross-platform method to map physical devices to partitions.
        """
        if platform.system() == "Windows":
            return self._map_physical_devices_windows()
        elif platform.system() == "Linux":
            return self._map_physical_devices_linux()
        else:
            raise NotImplementedError(f"Unsupported platform: {platform.system()}")

    def _map_physical_devices_windows(self):
        """
        Map physical devices to partitions using WMI on Windows.
        Parse device names and convert them to lowercase for consistency.
        """
        if wmi is None:
            raise ImportError("WMI is required on Windows for this functionality. Please install the `wmi` module.")

        disk_map = {}
        c = wmi.WMI()

        # Query Win32_DiskDrive to get physical disks
        for disk in c.Win32_DiskDrive():
            disk_name = disk.DeviceID.lower()  # Convert to lowercase, e.g., '\\.\physicaldrive0'
            parsed_disk_name = disk_name.replace("\\\\.\\", "")  # Parse to 'physicaldrive0'

            # Find the partitions associated with this disk
            partitions = []
            for partition in disk.associators("Win32_DiskDriveToDiskPartition"):
                for logical_disk in partition.associators("Win32_LogicalDiskToPartition"):
                    partitions.append(logical_disk.DeviceID)  # Convert to lowercase

            disk_map[parsed_disk_name] = partitions

        return disk_map

    def _map_physical_devices_linux(self):
        """
        Map physical devices to partitions on Linux by matching device prefixes.
        """
        disk_map = {}
        partitions = psutil.disk_partitions()
        io_counters = psutil.disk_io_counters(perdisk=True)

        for partition in partitions:
            device_name = partition.device  # e.g., /dev/sda1
            # Derive physical device from partition name (e.g., /dev/sda1 -> /dev/sda)
            if device_name.startswith("/dev/"):
                physical_device = device_name.rstrip("1234567890")  # Strip partition numbers
                if physical_device not in io_counters:
                    # Check for devices with "p" in their names, e.g., nvme0n1p1
                    physical_device = device_name.rstrip("p1234567890")
                if physical_device not in disk_map:
                    disk_map[physical_device] = []
                disk_map[physical_device].append(device_name)

        return disk_map

    def _aggregate_partition_usage(self, partitions):
        """
        Aggregate disk usage stats for all partitions belonging to a physical device.
        Returns total_space, free_space, used_space, percent_used.
        """
        total_space = free_space = used_space = percent_used = 0
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition)
                total_space += usage.total / 1024 ** 3  # Convert to GB
                free_space += usage.free / 1024 ** 3
                used_space += usage.used / 1024 ** 3
                percent_used += usage.percent
            except PermissionError:
                # Ignore inaccessible partitions
                pass

        # Average percent_used if we aggregated multiple partitions
        if len(partitions) > 0:
            percent_used /= len(partitions)

        return total_space, free_space, used_space, percent_used

    def get_report(self):
        return [info.state_dict() for info in self.info]

    def __repr__(self):
        return '\n'.join(repr(info) for info in self.info)

    def get_disk_index_for_path(self, path):
        """
        Given a path, return the index of the physical disk and its associated partition
        that the path belongs to.

        :param path: File system path (e.g., "C:\\path\\to\\file" on Windows or "/home/user" on Linux).
        :return: Tuple of (disk_index, partition) or (None, None) if not found.
        """
        abs_path = os.path.realpath(path)  # Get the absolute path

        for disk_index, disk_info in enumerate(self.info):  # Loop through all disks
            # Access the partitions from the static dictionary inside each DiskInfo object
            partitions = disk_info.static.get('partitions', [])
            for partition in partitions:  # Loop through all partitions of the disk
                try:
                    # Check if the path matches the mount point of the partition
                    partition_mount_point = os.path.realpath(partition)
                    common_path = os.path.commonpath([abs_path, partition_mount_point])

                    if common_path == partition_mount_point:  # Path belongs to this partition
                        return disk_index, partition
                except ValueError:
                    # Skip if paths are not on the same drive or if there's a mismatch
                    continue

        # Path does not belong to any known partition
        return None, None
