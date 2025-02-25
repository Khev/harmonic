import inspect
from agents.pg import PG
from agents.a2c import A2C  # Custom A2C implementation
from envs.env_gridworld import gridWorld

def filter_kwargs(func, kwargs):
    """Filter kwargs to only include those that match the function's signature."""
    signature = inspect.signature(func)
    valid_keys = signature.parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valid_keys}

def get_agent(agent, env, **kwargs):
    # Filter out 'agent', 'env', and 'env_name' from kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['agent', 'env', 'env_name']}
    if agent == 'pg':
        return PG(env, **filter_kwargs(PG.__init__, filtered_kwargs))
    elif agent == 'a2c':
        return A2C(env, **filter_kwargs(A2C.__init__, filtered_kwargs))
    else:
        raise ValueError(f"Unsupported agent type: {agent}")

def get_env(**kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['agent', 'env_name']}
    if kwargs.get('env_name') == 'grid_world':
        return gridWorld(**filter_kwargs(gridWorld.__init__, filtered_kwargs))
    else:
        raise ValueError(f"Unknown environment: {kwargs.get('env_name')}")


