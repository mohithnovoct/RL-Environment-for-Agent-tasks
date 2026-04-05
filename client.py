# Copyright (c) 2026. All rights reserved.
# Data Cleaning Environment - Client

"""
Data Cleaning Environment Client.

Provides client-side interface for connecting to a DataCleaningEnvironment server.
Supports async context manager, sync wrapper, and Docker container management.

Example (async):
    >>> async with DataCleaningEnv(base_url="http://localhost:8000") as env:
    ...     result = await env.reset()
    ...     result = await env.step(DataCleaningAction(
    ...         operation="fill_null", column="age", params={"strategy": "mean"}
    ...     ))
    ...     print(result.observation.data_quality_score)

Example (sync):
    >>> with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset()
    ...     result = env.step(DataCleaningAction(operation="trim_whitespace", column="name"))

Example (Docker):
    >>> env = await DataCleaningEnv.from_docker_image("data-cleaning-env:latest")
    >>> result = await env.reset()
    >>> await env.close()
"""

from openenv.core.env_client import EnvClient

from models import DataCleaningAction, DataCleaningObservation


class DataCleaningEnv(EnvClient):
    """Client for the Data Cleaning Environment.

    Inherits from EnvClient which provides:
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action
    - state(): Get current state
    - from_docker_image(): Start a Docker container
    - sync(): Get synchronous wrapper
    """
    pass
