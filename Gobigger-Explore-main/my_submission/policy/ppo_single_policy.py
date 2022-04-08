from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy
import numpy as np
from torch.distributions import Independent, Normal

from ding.torch_utils import Adam, to_device
from ding.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, get_gae_with_default_last_value, \
    v_nstep_td_data, v_nstep_td_error, get_nstep_return_data, get_train_sample, gae, gae_data, ppo_error_continuous, \
    get_gae
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, split_data_generator, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from ding.policy.base_policy import Policy
# from .common_utils import default_preprocess_learn
from ding.utils import dicts_to_lists, lists_to_dicts
import torch.nn.functional as F


def measure_before_split(data):
    my_data = copy.deepcopy(data)
    obs = my_data['obs']
    next_obs = my_data['next_obs']
    batch = my_data['obs']['batch']
    player_num_per_team = my_data['obs']['player_num_per_team']
    # data['obs']['clone'][0].numel()
    for k in obs:
        if k=='scalar' or k=='thorn_mask' or k=='clone_mask':
            try:
                obs[k] = obs[k].reshape(512, 3, obs[k].shape[1])
            except:
                print('*******************此处有错误【2】**********************\n')
        elif k=='food_relation' or k=='clone':
            try:
                obs[k] = obs[k].reshape(512, 3, obs[k].shape[1], obs[k].shape[2])
            except:
                print('*******************此处有错误【3】**********************\n')
        elif k=='food' or k=='thorn_relation' or k=='clone_relation':
            try:
                obs[k] = obs[k].reshape(512, 3, obs[k].shape[1], obs[k].shape[2], obs[k].shape[3])
            except:
                print('*******************此处有错误【4】**********************\n')
        elif k=='batch':
            obs[k] = my_data['obs']['batch']
        elif k=='player_num_per_team':
            obs[k] = my_data['obs']['player_num_per_team']
        else:
            print('*******************有未处理完的obs**********************\n')

    for k in next_obs:
        if k=='scalar' or k=='thorn_mask' or k=='clone_mask':
            try:
                next_obs[k] = next_obs[k].reshape(512, 3, next_obs[k].shape[1])
            except:
                print('*******************[2]此处有错误【2】**********************\n')
        elif k=='food_relation' or k=='clone':
            try:
                next_obs[k] = next_obs[k].reshape(512, 3, next_obs[k].shape[1], next_obs[k].shape[2])
            except:
                print('*******************[2]此处有错误【3】**********************\n')
        elif k=='food' or k=='thorn_relation' or k=='clone_relation':
            try:
                next_obs[k] = next_obs[k].reshape(512, 3, next_obs[k].shape[1], next_obs[k].shape[2],
                                                  next_obs[k].shape[3])
            except:
                print('*******************[2]此处有错误【4】**********************\n')
        elif k=='batch':
            next_obs[k] = my_data['next_obs']['batch']
        elif k=='player_num_per_team':
            next_obs[k] = my_data['next_obs']['player_num_per_team']
        else:
            print('*******************有未处理完的next_obs**********************\n')

    my_data['obs'] = obs
    my_data['next_obs'] = next_obs
    return my_data


def squeeze_after_split(batch_obs, split_size):
    obs = copy.deepcopy(batch_obs)
    # new_obs = {}
    for k in obs:
        if k=='batch' or k=='player_num_per_team':
            obs[k] = obs[k]
        else:
            try:
                obs[k] = obs[k].squeeze(0)
            except:
                print('******************* squeeze()数据有误 **********************\n')
            if k=='scalar' or k=='thorn_mask' or k=='clone_mask':
                obs[k] = obs[k].reshape(split_size*3, obs[k].shape[2])
            elif k=='food_relation' or k=='clone':
                obs[k] = obs[k].reshape(split_size*3, obs[k].shape[2], obs[k].shape[3])
            elif k=='food' or k=='thorn_relation' or k=='clone_relation':
                obs[k] = obs[k].reshape(split_size*3, obs[k].shape[2], obs[k].shape[3], obs[k].shape[4])
            else:
                print('-------------有数据没处理完-------------------')

    return obs


def gobigger_collate(data):
    r"""
    Arguments:
        - data (:obj:`list`): Lsit type data, [{scalar:[player_1_scalar, player_2_scalar, ...], ...}, ...]
    """
    B, player_num_per_team = len(data), len(data[0]['scalar'])
    data = {k: sum([d[k] for d in data], []) for k in data[0].keys() if not k.startswith('collate_ignore')}
    clone_num = max([x.shape[0] for x in data['clone']])
    thorn_num = max([x.shape[1] for x in data['thorn_relation']])
    food_h = max([x.shape[1] for x in data['food']])
    food_w = max([x.shape[2] for x in data['food']])
    data['scalar'] = torch.stack([torch.as_tensor(x) for x in data['scalar']]).float() # [B*player_num_per_team,5]
    data['food'] = torch.stack([F.pad(torch.as_tensor(x), (0, food_w - x.shape[2], 0, food_h - x.shape[1])) for x in data['food']]).float()
    data['food_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[0])) for x in data['food_relation']]).float()
    data['thorn_mask'] = torch.stack([torch.arange(thorn_num) < x.shape[1] for x in data['thorn_relation']]).float()
    data['thorn_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, thorn_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['thorn_relation']]).float()
    data['clone_mask'] = torch.stack([torch.arange(clone_num) < x.shape[0] for x in data['clone']]).float()
    data['clone'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[0])) for x in data['clone']]).float()
    data['clone_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['clone_relation']]).float()
    data['batch'] = B
    data['player_num_per_team'] = player_num_per_team
    return data


def default_preprocess_learn(
        data: List[Any],
        use_priority_IS_weight: bool = False,
        use_priority: bool = False,
        use_nstep: bool = False,
        ignore_done: bool = False,
) -> dict:

    # caculate max clone num
    tmp = [d['obs'] for d in data]
    tmp = {k: sum([d[k] for d in tmp], []) for k in tmp[0].keys() if not k.startswith('collate_ignore')}
    max_clone_num = max([x.shape[0] for x in tmp['clone']])
    limit = 52
    #print('max_clone_num:{}, limit:{}'.format(max_clone_num,limit))
    mini_bs = int(len(data)//2)
    if max_clone_num > limit:
        split_data1 = data[:mini_bs]
        split_data2 = data[mini_bs:]

        re = []
        for dt in (split_data1, split_data2):
            obs = [d['obs'] for d in dt]
            next_obs = [d['next_obs'] for d in dt]
            for i in range(len(dt)):
                dt[i] = {k: v for k, v in dt[i].items() if not 'obs' in k}
            dt = default_collate(dt)
            dt['obs'] = gobigger_collate(obs)
            dt['next_obs'] = gobigger_collate(next_obs)
            if ignore_done:
                dt['done'] = torch.zeros_like(dt['done']).float()
            else:
                dt['done'] = dt['done'].float()
            if use_priority_IS_weight:
                assert use_priority, "Use IS Weight correction, but Priority is not used."
            if use_priority and use_priority_IS_weight:
                dt['weight'] = dt['IS']
            else:
                dt['weight'] = dt.get('weight', None)
            if use_nstep:
                # Reward reshaping for n-step
                reward = dt['reward']
                if len(reward.shape) == 1:
                    reward = reward.unsqueeze(1)
                # reward: (batch_size, nstep) -> (nstep, batch_size)
                dt['reward'] = reward.permute(1, 0).contiguous()
            re.append(dt)
        return re

    # data collate
    obs = [d['obs'] for d in data]
    next_obs = [d['next_obs'] for d in data]
    for i in range(len(data)):
        data[i] = {k: v for k, v in data[i].items() if not 'obs' in k}

    data = default_collate(data)
    data['obs'] = gobigger_collate(obs)
    data['next_obs'] = gobigger_collate(next_obs)
    if ignore_done:
        data['done'] = torch.zeros_like(data['done']).float()
    else:
        data['done'] = data['done'].float()
    if use_priority_IS_weight:
        assert use_priority, "Use IS Weight correction, but Priority is not used."
    if use_priority and use_priority_IS_weight:
        data['weight'] = data['IS']
    else:
        data['weight'] = data.get('weight', None)
    if use_nstep:
        # Reward reshaping for n-step
        reward = data['reward']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        # reward: (batch_size, nstep) -> (nstep, batch_size)
        data['reward'] = reward.permute(1, 0).contiguous()
    return data


@POLICY_REGISTRY.register('gobigger_ppo')
class PPOPolicy(Policy):
    r"""
    Overview:
        Policy class of on policy version PPO algorithm.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppo',
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        # (bool) Whether to use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        recompute_adv=True,
        continuous=False,
        nstep_return=False,
        multi_agent=False,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=True,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            epoch_per_collect=10,
            # 10, 64
            batch_size=64,
            learning_rate=3e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.0,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=True,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            ignore_done=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.9,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
        ),
        eval=dict(),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main model.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in PPO"

        self._continuous = self._cfg.continuous
        if self._cfg.learn.ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            if self._continuous:
                # init log sigma
                if hasattr(self._model.actor_head, 'log_sigma_param'):
                    torch.nn.init.constant_(self._model.actor_head.log_sigma_param, -0.5)
                for m in list(self._model.critic.modules()) + list(self._model.actor.modules()):
                    if isinstance(m, torch.nn.Linear):
                        # orthogonal initialization
                        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        torch.nn.init.zeros_(m.bias)
                # do last policy layer scaling, this will make initial actions have (close to)
                # 0 mean and std, and will help boost performances,
                # see https://arxiv.org/abs/2006.05990, Fig.24 for details
                for m in self._model.actor.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.zeros_(m.bias)
                        m.weight.data.copy_(0.01 * m.weight.data)

        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.grad_clip_value
        )

        self._learn_model = model_wrap(self._model, wrapper_name='base')

        # Algorithm config
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm
        self._value_norm = self._cfg.learn.value_norm
        if self._value_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv
        # Main model
        self._learn_model.reset()

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac
        """
        # print('++++++++++++++++++++++++++++in _forward_learn ++++++++++++++++++++++++++++++++++++++++++')
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        if data['done'].max() > 0.5:
            print('-------------------I am here! ---------------------')
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):      # self._cfg.learn.epoch_per_collect : 10
            if self._recompute_adv:  # new v network compute new value
                with torch.no_grad():
                    value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                    next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                    if self._value_norm:
                        value *= self._running_mean_std.std     # self._running_mean_std.std: 1.000000005
                        next_value *= self._running_mean_std.std
                    # TODO:1
                    compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], data['traj_flag'])
                    data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)  # 0.9, 0.95

                    unnormalized_returns = value + data['adv']

                    if self._value_norm:
                        data['value'] = value / self._running_mean_std.std
                        data['return'] = unnormalized_returns / self._running_mean_std.std
                        self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                    else:
                        data['value'] = value
                        data['return'] = unnormalized_returns

            else:  # don't recompute adv
                if self._value_norm:
                    unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                    data['return'] = unnormalized_return / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_return.cpu().numpy())
                else:
                    data['return'] = data['adv'] + data['value']

            # split_data_generator() 前要将obs 与 next_obs 转换为 512 * 3 的结构再进行打乱顺序
            # my_data = measure_before_split(data)

            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                # batch['obs'] = squeeze_after_split(batch['obs'], self._cfg.learn.batch_size)
                output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')
                adv = batch['adv'].squeeze(-1)
                output['logit'] = output['logit'].squeeze(1)
                batch['logit'] = batch['logit'].squeeze(1)
                if self._adv_norm:
                    # Normalize advantage in a train_batch
                    adv = (adv - adv.mean()) / (adv.std() + 1e-2)

                # Calculate ppo error
                if self._continuous:
                    ppo_batch = ppo_data(
                        output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    ppo_loss, ppo_info = ppo_error_continuous(ppo_batch, self._clip_ratio)
                else:
                    ppo_batch = ppo_data(
                        output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    # TODO: 2
                    ppo_loss, ppo_info = ppo_error(ppo_batch, self._clip_ratio)
                wv, we = self._value_weight, self._entropy_weight
                total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss
                total_loss = total_loss.clamp(-5.0, 5.0)
                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': total_loss.item(),
                    'policy_loss': ppo_loss.policy_loss.item(),
                    'value_loss': ppo_loss.value_loss.item(),
                    'entropy_loss': ppo_loss.entropy_loss.item(),
                    'adv_max': adv.max().item(),
                    'adv_mean': adv.mean().item(),
                    'value_mean': output['value'].mean().item(),
                    'value_max': output['value'].max().item(),
                    'approx_kl': ppo_info.approx_kl,
                    'clipfrac': ppo_info.clipfrac,
                    'act': batch['action'].float().mean().item(),
                    'reward_mean':batch['reward'].float().mean().item(),
                }
                if self._continuous:
                    return_info.update(
                        {
                            'mu_mean': output['logit'][0].mean().item(),
                            'sigma_mean': output['logit'][1].mean().item(),
                        }
                    )
                return_infos.append(return_info)
            # print('**************epoch*-----------------')
        return return_infos

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._continuous = self._cfg.continuous
        if self._continuous:
            self._collect_model = model_wrap(self._model, wrapper_name='base')
        else:
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = gobigger_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor_critic')
            if self._continuous:
                (mu, sigma), value = output['logit'], output['value']
                dist = Independent(Normal(mu, sigma), 1)
                output['action'] = dist.sample()
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        """
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation
                - model_output (:obj:`dict`): Output of collect model, including at least ['action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'logit': model_output['logit'],
            'value': model_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and calculate GAE, return one data to cache for next time calculation
        Arguments:
            - data (:obj:`list`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = to_device(data, self._device)
        for transition in data:
            transition['traj_flag'] = copy.deepcopy(transition['done'])
        data[-1]['traj_flag'] = True

        if self._cfg.learn.ignore_done:
            data[-1]['done'] = False

        if data[-1]['done']:
            last_value = torch.zeros_like(data[-1]['value'])
        else:
            data_input = gobigger_collate(list([data[-1]['next_obs']]))
            data_input = to_device(data_input, self._device)
            if self._cfg.multi_agent:
                with torch.no_grad():
                    last_value = self._collect_model.forward(
                        {
                            'agent_state': data[-1]['next_obs']['agent_state'].unsqueeze(0),
                            'global_state': data[-1]['next_obs']['global_state'].unsqueeze(0),
                            'action_mask': data[-1]['next_obs']['action_mask'].unsqueeze(0)
                        },
                        mode='compute_actor_critic'
                    )['value']
                    last_value = last_value.squeeze(0)
            else:
                with torch.no_grad():
                    last_value = self._collect_model.forward(
                        data_input, mode='compute_actor_critic'     # thorn_mask 和 clone_mask 不是cuda类型
                    )['value']
                    last_value = last_value.squeeze(0)
        if self._value_norm:
            last_value *= self._running_mean_std.std
            for i in range(len(data)):
                data[i]['value'] *= self._running_mean_std.std
        data = get_gae(
            data,
            to_device(last_value, self._device),
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=True,
        )
        if self._value_norm:
            for i in range(len(data)):
                data[i]['value'] /= self._running_mean_std.std

        # remove next_obs for save memory when not recompute adv
        if not self._recompute_adv:
            for i in range(len(data)):
                data[i].pop('next_obs')
        return get_train_sample(data, self._unroll_len)


    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._continuous = self._cfg.continuous
        if self._continuous:
            self._eval_model = model_wrap(self._model, wrapper_name='base')
        else:
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = gobigger_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
            if self._continuous:
                (mu, sigma) = output['logit']
                output.update({'action': mu})
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        if self._cfg.multi_agent:
            return 'mappo', ['ding.model.template.mappo']
        else:
            return 'vac', ['ding.model.template.vac']

    def _monitor_vars_learn(self) -> List[str]:
        variables = super()._monitor_vars_learn() + [
            'policy_loss',
            'value_loss',
            'entropy_loss',
            'adv_max',
            'adv_mean',
            'approx_kl',
            'clipfrac',
            'value_max',
            'value_mean',
            'reward_mean'
        ]
        if self._continuous:
            variables += ['mu_mean', 'sigma_mean', 'sigma_grad', 'act']
        return variables
