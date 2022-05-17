import inspect
import functools
import torch


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def td_lambda_target(batch, max_episode_len, q_targets, args):
    # batch.shep = (episode_num, max_episode_len, n_agents, n_actions)
    # q_targets.shape = (episode_num, max_episode_len, n_agents)
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float()).repeat(1, 1, args.n_agents)
    terminated = (1 - batch["terminated"].float()).repeat(1, 1, args.n_agents)
    r = batch['r'].repeat((1, 1, args.n_agents))
    # --------------------------------------------------n_step_return---------------------------------------------------
    '''
    1. Each experience has several n_step_returns, so give a maximum max_episode_len dimension to install n_step_return
    The last dimension, the nth number represents n+1 step.
    2. Because the length of each episode in the batch is different, you need to use mask to set the extra n-step return to 0,
    Otherwise, it will affect the subsequent lambda return. The lambda return of the t-th experience is related to all n-step returns after it,
    If it is not set to 0, it is too late to set 0 after calculating td-error.
    3. terminated is used to set q_targets and r that exceed the length of the current episode to 0.
    '''
    n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))
    for transition_idx in range(max_episode_len - 1, -1, -1):
        # Finally calculate 1 step return
        n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]) * mask[:, transition_idx]       
        # Also note that the index corresponding to n step return is n-1
        for n in range(1, max_episode_len - transition_idx):
            # n step return at time t =r + gamma * (n-1 step return at time t + 1)
            # Except for n=1, 1 step return =r + gamma * (Q at time t + 1)
            n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, :, n - 1]) * mask[:, transition_idx]
    # --------------------------------------------------n_step_return---------------------------------------------------

    # --------------------------------------------------lambda return---------------------------------------------------
    '''
    lambda_return.shape = (episode_num, max_episode_len, n_agents)
    '''
    lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))
    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num, args.n_agents))
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
        lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                           pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                           n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
    # --------------------------------------------------lambda return---------------------------------------------------
    return lambda_return
