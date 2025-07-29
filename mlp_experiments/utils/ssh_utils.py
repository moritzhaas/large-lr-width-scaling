import h5py
import numpy as np
import paramiko
import time

def get_recent_files_via_ssh(ssh, directory, extension='.hdf5', num_files=1):
    """
    Get the most recent files in a directory with a specific extension via SSH.
    
    Args:
        ssh (paramiko.SSHClient): SSH client object.
        directory (str): Directory to search for files.
        extension (str): File extension to filter by.
        num_files (int): Number of recent files to return.
    
    Returns:
        list: List of recent files.
    """
    # List all files in the directory with the specified extension
    stdin, stdout, stderr = ssh.exec_command(f"ls -t {directory}/*{extension}")
    files = stdout.read().decode().split()
    
    # Return the most recent files
    return files[:num_files]

def get_recent_folders_via_ssh(ssh, directory, num_folders=1):
    """
    Get the most recent folders in a directory via SSH.
    
    Args:
        ssh (paramiko.SSHClient): SSH client object.
        directory (str): Directory to search for folders.
        num_folders (int): Number of recent folders to return.
    
    Returns:
        list: List of recent folders.
    """
    # List all folders in the directory
    stdin, stdout, stderr = ssh.exec_command(f"ls -dt {directory}/*/")
    folders = stdout.read().decode().split()
    
    # Return the most recent folders
    return folders[:num_folders]


def read_hdf5_keys_via_ssh(ssh, filename):
    """
    Read keys from an HDF5 file via SSH.
    
    Args:
        ssh (paramiko.SSHClient): SSH client object.
        filename (str): HDF5 filename.
    
    Returns:
        list: List of keys in the HDF5 file.
    """
    sftp = ssh.open_sftp()
    with sftp.open(filename, 'rb') as f:
        with h5py.File(f, 'r') as hdf5_file:
            keys = list(hdf5_file.keys())
            subkeys = list(hdf5_file[keys[0]].keys()) if isinstance(hdf5_file[keys[0]], h5py.Group) else hdf5_file[keys[0]][()]
    sftp.close()
    return keys, subkeys

def read_hdf5_entry_via_ssh(ssh, filename, entry_key):
    """
    Read a single entry from HDF5 file by its outer key via SSH.
    
    Args:
        ssh (paramiko.SSHClient): SSH client object.
        filename (str): Input HDF5 filename
        entry_key (int): The outer key to load (e.g., 17)
    
    Returns:
        dict: Single inner dictionary corresponding to entry_key
        None: If entry_key doesn't exist
    """
    sftp = ssh.open_sftp()
    with sftp.open(filename, 'rb') as f:
        with h5py.File(f, 'r') as hdf5_file:
            # Convert key to string for HDF5 lookup
            key_str = str(entry_key)
            
            # Check if key exists
            if key_str not in hdf5_file:
                return None
                
            # Read just this group
            inner_dict = {}
            for key in hdf5_file[key_str]:
                value = hdf5_file[key_str][key][()]
                # Convert numpy types back to Python native types
                if isinstance(value, np.generic):
                    value = value.item()
                inner_dict[key] = value
    sftp.close()
    return inner_dict

def get_stats_from_h5py_via_ssh_old(ssh, filename, statname):
    # for convoluted way of saving stats: [iter][statname]
    stats = {}
    sftp = ssh.open_sftp()
    with sftp.open(filename, 'rb') as f:
        with h5py.File(f, 'r') as hdf5_file:
            keys = list(hdf5_file.keys())
            for key in keys:
                if statname in hdf5_file[key]:
                    # if isinstance(value, np.generic):
                    #     value = value.item()
                    stats[key] = hdf5_file[key][statname][()]
    sftp.close()
    return stats

def get_stats_from_h5py_via_ssh(ssh, filename, statname):
    # TODO: for corrected way of saving stats
    return None



def establish_ssh_connection(hostname, port, username, password, retries=3, delay=5):
    """
    Establish an SSH connection with retry logic.
    
    Args:
        hostname (str): SSH server hostname.
        port (int): SSH server port.
        username (str): SSH username.
        password (str): SSH password.
        retries (int): Number of retry attempts.
        delay (int): Delay between retries in seconds.
    
    Returns:
        paramiko.SSHClient: Established SSH client.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    for attempt in range(retries):
        try:
            ssh.connect(hostname, port, username, password)
            print("SSH connection established.")
            return ssh
        except paramiko.SSHException as e:
            print(f"SSH connection failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e
            

def execute_with_retries(func, ssh, ssh_config, *args, retries=3, delay=5, **kwargs):
    """
    Execute a function with retry logic for SSH-dependent operations.
    
    Args:
        func (callable): Function to execute.
        ssh (paramiko.SSHClient): SSH client object.
        retries (int): Number of retry attempts.
        delay (int): Delay between retries in seconds.
    
    Returns:
        Any: Result of the function execution.
    """
    for attempt in range(retries):
        try:
            return func(ssh, *args, **kwargs)
        except (paramiko.SSHException, IOError) as e:
            print(f"Operation failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                hostname = ssh_config['ssh']['hostname']
                port = int(ssh_config['ssh']['port'])
                username = ssh_config['ssh']['username']
                password = ssh_config['ssh']['password']
                ssh = establish_ssh_connection(hostname, port, username, password)
            else:
                raise e