import os
import platform
import psutil
import time

# Import WMI for Windows
try:
    import wmi
except ImportError:
    wmi = None


class DiskInfo:
    def __init__(self, name, total_space, free_space, used_space, percent_used, read_rate=None, write_rate=None,
                 partitions=None):
        self.name = name
        self.total_space = total_space
        self.free_space = free_space
        self.used_space = used_space
        self.percent_used = percent_used
        self.read_rate = read_rate
        self.write_rate = write_rate
        self.partitions = partitions or []

    def state_dict(self):
        return {
            'name': self.name,
            'total_space': self.total_space,
            'free_space': self.free_space,
            'used_space': self.used_space,
            'percent_used': self.percent_used,
            'read_rate': self.read_rate,
            'write_rate': self.write_rate,
            'partitions': self.partitions,
        }

    def __repr__(self):
        partition_list = ", ".join(self.partitions)
        return (
            f"Disk {self.name}: Total Space: {self.total_space:.2f} GB, "
            f"Free Space: {self.free_space:.2f} GB, "
            f"Used Space: {self.used_space:.2f} GB, "
            f"Percent Used: {self.percent_used:.2f}%, "
            f"Read Rate: {self.read_rate:.2f} MB/s, "
            f"Write Rate: {self.write_rate:.2f} MB/s, "
            f"Partitions: [{partition_list}]"
        )


class DiskReport:
    def __init__(self, interval=0.1, num_samples=5):
        self.interval = interval
        self.num_samples = num_samples
        self.info = self.make_info()

    def make_info(self):
        # Get the physical device-to-partition mapping
        physical_device_map = self._get_physical_device_map()
        # Initialize dictionaries to track read/write rates for physical devices
        disk_samples = {device.lower(): {'read_bytes': [], 'write_bytes': []} for device in
                        psutil.disk_io_counters(perdisk=True)}
        disk_info_list = []

        try:
            # Sampling loop
            for _ in range(self.num_samples):
                io_counters = psutil.disk_io_counters(perdisk=True)
                for device, stats in io_counters.items():
                    if device.lower() in disk_samples:
                        disk_samples[device.lower()]['read_bytes'].append(stats.read_bytes / 1024 ** 2)  # MB
                        disk_samples[device.lower()]['write_bytes'].append(stats.write_bytes / 1024 ** 2)  # MB

                time.sleep(self.interval)

            # Process each physical device
            for physical_device, partitions in physical_device_map.items():
                # Calculate read/write rates
                samples = disk_samples.get(physical_device, {'read_bytes': [], 'write_bytes': []})
                read_rate = self.calculate_rate(samples['read_bytes'])
                write_rate = self.calculate_rate(samples['write_bytes'])

                # Combine space information from partitions
                total_space, free_space, used_space, percent_used = self._aggregate_partition_usage(partitions)

                # Create DiskInfo object
                disk_info = DiskInfo(
                    name=physical_device,
                    total_space=total_space,
                    free_space=free_space,
                    used_space=used_space,
                    percent_used=percent_used,
                    read_rate=read_rate,
                    write_rate=write_rate,
                    partitions=partitions,
                )
                disk_info_list.append(disk_info)

        except Exception as e:
            raise ValueError(f"Failed to retrieve Disk information: {str(e)}")

        return disk_info_list

    def calculate_rate(self, samples):
        if len(samples) < 2:
            return 0
        rates = [(samples[i] - samples[i - 1]) / self.interval for i in range(1, len(samples))]
        return sum(rates) / len(rates)

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
            for partition in disk_info.partitions:  # Loop through all partitions of the disk
                try:
                    # Check if the path matches the mount point of the partition
                    partition_mount_point = os.path.realpath(partition)
                    common_path = os.path.commonpath([abs_path, partition_mount_point])

                    if common_path == partition_mount_point:  # Path belongs to this partition
                        return disk_info, partition
                except ValueError:
                    # Skip if paths are not on the same drive or if there's a mismatch
                    continue

        # Path does not belong to any known partition
        return None, None
