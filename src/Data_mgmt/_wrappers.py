from loguru import logger
import time
from typing import Callable

from sshtunnel import SSHTunnelForwarder


def _check_input(mandatory: list, **kwargs):
    for elem in mandatory:
        if elem not in kwargs:
            raise ValueError(f"Mandatory keyword argument {elem} omitted")


def time_debug(func: Callable):
    """Decorator that reports the execution time."""

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        logger.debug(f"{func.__name__} execution took {end - start}")
        return result

    return wrap


def ssh_tunnel(fun):
    """Decorator for managing ssh tunnel in function call

    Mandatory keyword args:
       | ssh_host (str): host ssh server
       | ssh_user (str): ssh username
       | ssh_pkey (str): path to ssh private key

    Optional keyword args:
       | local_port (int): - local binding port 5432 by default
       | remote_port (int): - remote binding port 22 by default
       | local_address (str): - local bind address '127.0.0.1' by default

    important: include any *args, **kwargs that are used by decorated function!
    """

    def wrap(*args, **kwargs):
        _check_input(['ssh_host', 'ssh_user', 'ssh_pkey'], **kwargs)

        ssh_host: str = kwargs.pop('ssh_host')
        ssh_user: str = kwargs.pop('ssh_user')
        ssh_pkey: str = kwargs.pop('ssh_pkey')

        local_port: int = kwargs.pop('local_port') if 'local_port' in kwargs else 5432
        remote_port: int = kwargs.pop('remote_port') if 'remote_port' in kwargs else 22
        local_address: str = kwargs.pop('local_address') if 'local_address' in kwargs else '127.0.0.1'

        with SSHTunnelForwarder(
                (ssh_host, remote_port),
                ssh_username=ssh_user,
                ssh_pkey=ssh_pkey,
                remote_bind_address=(local_address, local_port)
        ) as server:
            server.start()  # start ssh sever
            logger.info("Server connected via ssh")

            kwargs['ssh_server'] = server  # give reference of server to decorated fun
            result = fun(*args, **kwargs)

        return result

    return wrap
