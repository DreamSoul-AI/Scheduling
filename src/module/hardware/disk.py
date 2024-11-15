import logging
import psutil
import os


class DiskInfo:
    def __init__(self, index, name, total_space, free_space, used_space, percent_used):
        self.index = index
        self.name = name
        self.total_space = total_space
        self.free_space = free_space
        self.used_space = used_space
        self.percent_used = percent_used

    def state_dict(self):
        return {
            'index': self.index,
            'name': self.name,
            'total_space': self.total_space,
            'free_space': self.free_space,
            'used_space': self.used_space,
            'percent_used': self.percent_used
        }

    def __repr__(self):
        return (
            f'Disk {self.index} ({self.name}): '
            f'Total Space: {self.total_space:.2f} GB, '
            f'Free Space: {self.free_space:.2f} GB, '
            f'Used Space: {self.used_space:.2f} GB, '
            f'Percent Used: {self.percent_used:.2f}%'
        )


class DiskReport:
    def __init__(self):
        self.info = self.make_info()

    def make_info(self):
        info = []
        try:
            partitions = psutil.disk_partitions()

            for i, partition in enumerate(partitions):
                if partition.fstype:  # Ignore non-mountable partitions
                    disk_usage = psutil.disk_usage(partition.mountpoint)

                    info_i = DiskInfo(
                        index=i,
                        name=partition.device,
                        total_space=disk_usage.total / 1024 ** 3,  # Convert to GB
                        free_space=disk_usage.free / 1024 ** 3,
                        used_space=disk_usage.used / 1024 ** 3,
                        percent_used=disk_usage.percent
                    )
                    info.append(info_i)
        except Exception as e:
            logger = logging.getLogger('main')
            logger.error(f'Failed to retrieve disk information: {str(e)}')
            return []

        return info

    def get_report(self):
        return [info.state_dict() for info in self.info]

    def __repr__(self):
        return '\n'.join(repr(info) for info in self.info)

    def get_disk_index_for_path(self, path):
        """Given a path, return the index of the disk partition it belongs to."""
            # Get the real (canonical) absolute path
        abs_path = os.path.realpath(path)

        # Find the partition that the path belongs to
        for partition in self.info:
            # Get the real mount point for the partition
            partition_mount_point = os.path.realpath(partition.name)

            # Use os.path.commonpath() to find the common path between the given path and the partition mount point
            try:
                common_path = os.path.commonpath([abs_path, partition_mount_point])
                # If the common path is the same as the partition's mount point, the path is inside this partition
                if common_path == partition_mount_point:
                    return partition
            except ValueError:
                # Catch the ValueError when the paths don't have the same drive
                continue
        # If no matching partition is found
        return


