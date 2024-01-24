from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import os
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field

from osrl.algorithms import CDT, CDTTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
from examples.configs.cdt_configs import CDT_DEFAULT_CONFIG, CDTTrainConfig
import types


# @dataclass
# class EvalConfig:
#     # path: str = "log/.../checkpoint/model.pt"
#     task: str = 'none'
#     exp: str = 'tmp'
#     returns: List[float] = field(default=[30, 30, 30], is_mutable=True)
#     costs: List[float] = field(default=[20, 40, 80], is_mutable=True)
#     noise_scale: List[float] = None
#     eval_episodes: int = 10
#     best: bool = False
#     device: str = "cpu"
#     threads: int = 4


@pyrallis.wrap()
def eval(args: CDTTrainConfig):

    # update config
    cfg, old_cfg = asdict(args), asdict(CDTTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(CDT_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    load_path = os.getcwd()+f'/save/{args.task}/{args.exp}'
    # cfg, _ = load_config_and_model(load_path, args.best)
    import pickle
    with open(load_path+'/cost_mean.pickle', 'rb') as f:
        data = pickle.load(f)
    costs_mean = data['cost_mean']
    seed_all(cfg["seed"])
    # args.device = "cpu"
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    #if "Metadrive" in cfg["task"]:
    #    import gym
    # else:
    import gymnasium as gym  # noqa

    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
    # env.set_target_cost(cfg["cost_limit"])
    env.set_target_cost(costs_mean)

    target_entropy = -env.action_space.shape[0]

    # model & optimizer & scheduler setup
    cdt_model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=cfg["embedding_dim"],
        seq_len=cfg["seq_len"],
        episode_len=cfg["episode_len"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        attention_dropout=cfg["attention_dropout"],
        residual_dropout=cfg["residual_dropout"],
        embedding_dropout=cfg["embedding_dropout"],
        time_emb=cfg["time_emb"],
        use_rew=cfg["use_rew"],
        use_cost=cfg["use_cost"],
        cost_transform=cfg["cost_transform"],
        add_cost_feat=cfg["add_cost_feat"],
        mul_cost_feat=cfg["mul_cost_feat"],
        cat_cost_feat=cfg["cat_cost_feat"],
        action_head_layers=cfg["action_head_layers"],
        cost_prefix=cfg["cost_prefix"],
        stochastic=cfg["stochastic"],
        init_temperature=cfg["init_temperature"],
        target_entropy=target_entropy,
    )
    # cdt_model.load_state_dict(model["model_state"])
    cdt_model.load_state_dict(torch.load(load_path+f'/model.pt'))
    cdt_model.eval()
    cdt_model.to(args.device)

    trainer = CDTTrainer(cdt_model,
                         env,
                         reward_scale=cfg["reward_scale"],
                         cost_scale=cfg["cost_scale"],
                         cost_reverse=cfg["cost_reverse"],
                         device=args.device)

    # rets = args.returns
    # costs = args.costs
    # assert len(rets) == len(
    #     costs
    # ), f"The length of returns {len(rets)} should be equal to costs {len(costs)}!"
    target_rets, target_costs = [], []
    for target_return in args.target_returns:
        reward_return, cost_return = target_return
        target_rets.append(reward_return)
        target_costs.append(cost_return)
        
    rets, costs = [], []
    rets_norm, costs_norm = [], []
    rets_prom, costs_prom = [], []
    rets_prom_norm, costs_prom_norm = [], []
    with torch.no_grad():
        for target_ret, target_cost in zip(target_rets, target_costs):
            ret, cost, length = trainer.evaluate(args.eval_episodes,
                                                target_ret * cfg["reward_scale"],
                                                target_cost * cfg["cost_scale"],
                                                prom=False)
            ret_prom, cost_prom, length_prom = trainer.evaluate(args.eval_episodes,
                                                target_ret * cfg["reward_scale"],
                                                target_cost * cfg["cost_scale"],
                                                prom=True)
            normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
            normalized_ret_prom, normalized_cost_prom = env.get_normalized_score(ret_prom, cost_prom)
            rets.append(ret); costs.append(cost), rets_norm.append(normalized_ret); costs_norm.append(normalized_cost)
            rets_prom.append(ret_prom); costs_prom.append(cost_prom), rets_prom_norm.append(normalized_ret_prom); costs_prom_norm.append(normalized_cost_prom)
    ret = sum(rets)/len(rets); cost = sum(costs)/len(costs); normalized_ret = sum(rets_norm)/len(rets_norm); normalized_cost = sum(costs_norm)/len(costs_norm)
    ret_prom = sum(rets_prom)/len(rets_prom); cost_prom = sum(costs_prom)/len(costs_prom); normalized_ret_prom = sum(rets_prom_norm)/len(rets_prom_norm); normalized_cost_prom = sum(costs_prom_norm)/len(costs_prom_norm)
    print(
        f"Target reward {target_ret}, real reward      {ret:.3f}, normalized reward:      {normalized_ret:.3f},      real cost {cost:.1f}, normalized cost: {normalized_cost:.3f}\n" 
        f"Target cost   {target_cost:.1f}, real reward prom {ret_prom:.3f}, normlalized_reward porm {normalized_ret_prom:.3f}, real cost prom {cost_prom:.1f}, normalized cost prom {normalized_cost_prom:.3f}"
    )


if __name__ == "__main__":
    eval()
