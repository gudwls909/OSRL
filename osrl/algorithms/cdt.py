from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from torch.distributions.beta import Beta
from torch.nn import functional as F  # noqa
from tqdm.auto import trange  # noqa

from osrl.common.net import DiagGaussianActor, TransformerBlock, mlp

import logging
import warnings
import inspect
from copy import deepcopy
import transformers
from ray import tune
import ray
from ray import air

import ray.train
import logging
import tempfile
import os
import pickle
from ray.train import Checkpoint

ray_data_logger = logging.getLogger("ray.data")
ray_tune_logger = logging.getLogger("ray.tune")
ray_rllib_logger = logging.getLogger("ray.rllib")
ray_train_logger = logging.getLogger("ray.train")
ray_serve_logger = logging.getLogger("ray.serve")

ray_data_logger.setLevel(logging.FATAL)
ray_tune_logger.setLevel(logging.FATAL)
ray_rllib_logger.setLevel(logging.FATAL)
ray_train_logger.setLevel(logging.FATAL)
ray_serve_logger.setLevel(logging.FATAL)

from torch.distributions.normal import Normal
import torch.nn.functional as F

from enum import Enum
logger = logging.getLogger(__name__)
logger.setLevel(logging.FATAL)


def generate_parallel2(model,
                inputs,
                generation_config=None,
                stopping_criteria=None,
                num_samples=1,
                **kwargs):
    
    wm = ray.put(model)
    
    @ray.remote(num_cpus=10, num_gpus=1)
    def run_model(model_):
        out = model_.generate(inputs, generation_config, stopping_criteria, **kwargs)
        # checkpoint_dict = {"output": out}
        # checkpoint: ray.train.Checkpoint = ray.tain.get_checkpoint()
        # if checkpoint:
        #     with checkpoint.as_directory() as temp_checkpoint_dir:
        #         check_file = os.path.join(temp_checkpoint_dir, "result.pkl")
        #         pickle.dump(checkpoint_dict, check_file)

        return out
                # ray.train.report({"metric":0},
                #                     checkpoint=checkpoint.from_directory(temp_checkpoint_dir))
            

    # os.environ["RAY_AIR_NEW_OUTPUT"] ="0"
    # run_with_resources = tune.with_resources(run_model, {'cpu':3, 'gpu':0.5})
    # run_config = air.RunConfig(verbose=0)
    
    # tuner = tune.Tuner(run_with_resources.remote(),
    #                     tune_config=tune.TuneConfig(num_samples=num_samples),
    #                     run_config=run_config)
    # results = tuner.fit()
    result_ids = [run_model.remote(wm) for _ in range(num_samples)]
    arr = []
    while len(result_ids):
    # results = ray.get(result_ids)
        done_id, result_ids = ray.wait(result_ids)
        res = ray.get(done_id[0])
        arr.append(res)
    return arr
    # r = ray.get(results)
    # ray_results = ray.wait(results)
    # for i in range(len(arr)):
    #     r = arr[i]
    #     with open(os.path.join(r), 'rb') as f:
    #         result_data = pickle.load(f)
    #     arr.append(result_data['output'])
    # return arr


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    GREEDY_SEARCH_WITH_OM = "greedy_search_with_om"
    GREEDY_SEARCH_WITH_OM_NO_PADDING = "greedy_search_with_om_no_padding"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


class GenerationMixin:
    def prepare_inputs_for_genearation(self, input_ids, **model_kwargs):
            # forward_args = inspect.getargspec(self.forward).args
            # states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
            if not isinstance(input_ids, dict):
                raise TypeError
            else:
                pass

            # ii = deepcopy(input_ids)                  

            # if self.max_length is not None:
            #     for k, v in ii.items():
            #         ii[k] = v[:,-self.max_length:]

            states = input_ids['states'][:,-self.seq_len:].detach()
            actions = input_ids['actions'][:,-self.seq_len:].detach()
            returns_to_go = input_ids['returns_to_go'][:,-self.seq_len:][:,:,0].detach()
            costs_to_go = input_ids['costs_to_go'][:,-self.seq_len:][:,:,0].detach()
            
            timesteps = torch.arange(0, states.shape[1]).to(device=states.device).reshape(1,-1)
            # ii.update(timesteps=timesteps)

            # attention_mask = torch.cat([torch.zeros(max(0,self.seq_len-states.shape[1])), torch.ones(states.shape[1])])
            # attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            # # states = ii['states']
            # # actions = ii['actions']
            # # returns_to_go = ii['returns_to_go']
            # # timesteps = ii['timesteps']
            
            
            states = torch.cat(
                [torch.zeros((states.shape[0], max(0,self.seq_len-states.shape[1]), self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], max(0,self.seq_len - actions.shape[1]), self.action_dim),
                                device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max(0,self.seq_len-returns_to_go.shape[1])), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            costs_to_go = torch.cat(
                [torch.zeros((costs_to_go.shape[0], max(0,self.seq_len-costs_to_go.shape[1])), device=costs_to_go.device), costs_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], max(0,self.seq_len-timesteps.shape[1])), device=timesteps.device), timesteps],
                dim=1
                ).to(dtype=torch.long)

            out_dict =dict(states=states,
                           actions=actions,
                           returns_to_go=returns_to_go,
                           costs_to_go=costs_to_go,
                           timesteps=timesteps,)

            # input_ids.update(states=states)
            # input_ids.update(actions=actions)
            # input_ids.update(returns_to_go=returns_to_go)
            # input_ids.update(timesteps=timesteps)
            # input_ids.update(rewards=rewards)
            return out_dict

    def _get_stopping_criteria(self, generation_config, stopping_criteria):
        # len1 = 
        # if len1 is None:
        #     sc = stopping_criteria
        # else:
        #     sc = min(len1, stopping_criteria)
        sc = stopping_criteria
        def stop_func(input_ids):
            seq_len = input_ids['states'].shape[1]
            return seq_len >= sc + generation_config
        return stop_func

    @torch.no_grad()
    def generate(self,
                 inputs,
                 generation_config=None,
                 stopping_criteria=None,
                 **kwargs):
        # states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        self.generation_config = generation_config
        model_kwargs = kwargs
        if stopping_criteria == None:
            stopping_criteria = self.episode_len


        # 2. set logit and stopping criteria

        # 3. define model inputs
        # input_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        #     inputs, generation_config)
        
        # 4. Define model kwargs

        attention_mask = None
        # attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
        # attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

        # 5. input id?
        input_ids = inputs
        prompt_len = input_ids["states"].size(1) 
        #inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 6. Prepare max_length depending on stopping criteria.
        # input_ids_length = input_ids.shape[-1]

        # 7. determine gernation mode
        if generation_config==None:
            generation_mode = GenerationMode.GREEDY_SEARCH
        else:
            generation_mode = generation_config["output_mode"]

        # 8. prepare distribution pre_processing samplers

        # 9. prepare stopping crieria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=prompt_len, stopping_criteria=stopping_criteria
        )

        # 10. go into generation mode
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            return self.greedy_search(input_ids,
                stopping_criteria=prepared_stopping_criteria,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.GREEDY_SEARCH_WITH_OM_NO_PADDING:
            return self.greedy_search_for_ocuppancy_measure(input_ids,
                stopping_criteria=prepared_stopping_criteria,
                **model_kwargs,
            )
        else:
            raise ValueError
        
    @torch.no_grad()    
    def greedy_search_for_ocuppancy_measure(self,
                      input_ids,
                      stopping_criteria,
                      **model_kwargs):
        
        # unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        om_arr = []
        # model_kwargs['max_k'] = 3
        while True:
            model_inputs = self.prepare_inputs_for_genearation(input_ids, **model_kwargs)
            # del a
            # states, actions, rewards, returns_to_go, timesteps,
        

            # forward pass to get next token
            # outputs = self(
            #     **model_inputs,
            #     attention_mask=attention_mask
            # )
            # print(model_inputs['states'][:,-20:])
            outputs = self.forward_with_om_no_padding(**model_inputs,
                                max_k=model_kwargs['max_k'],)
            # outputs = self.forward(**model_inputs, deterministic=False, attention_mask=attention_mask)
            # out_dicts = {'states': outputs[0].detach(),
            #              'actions': outputs[1].detach(),
            #              'returns_to_go': outputs[2].detach()} #, 'log_p_pi': outputs[3]}
            # om_arr.append(outputs[3].detach())

            out_dicts = {'states': outputs[0].detach(),
                         'actions': outputs[1].detach(),
                         'returns_to_go': outputs[2].detach(),
                         'costs_to_go': outputs[3].detach()} #, 'log_p_pi': outputs[3]}
            om_arr.append(outputs[4].detach())


            # del model_inputs, outputs
            # outputs: state_preds, action_preds, return_preds

            # next_token_logits = outputs.logits[:, -1, :]
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # store scores, atttenstions and hidden states when required?
            
            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            # print(out_dicts['states'].size(), input_ids['states'].size())

            # output_ids = {k: torch.cat([v, torch.unsqueeze(v[:,-1,...], 1)], dim=1) for k, v in out_dicts.items()}
            input_ids.update({k: torch.cat([v, torch.unsqueeze(out_dicts[k][:,-1,...], 1)], dim=1) for k, v in input_ids.items()})

            # model_kwargs = self._update_kwargs_for_generation(
            #     outputs, model_kwargs
            # )

            # stop if we exceed the max length
            if stopping_criteria(input_ids):
                this_peer_finished = True
                break
            torch.cuda.empty_cache()
        input_ids.update(occupancy_measure=torch.cat(om_arr, dim=0).unsqueeze(0))

        return input_ids
        




class CDT(nn.Module, GenerationMixin):
    """
    Constrained Decision Transformer (CDT)
    
    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        seq_len (int): The length of the sequence to process.
        episode_len (int): The length of the episode.
        embedding_dim (int): The dimension of the embeddings.
        num_layers (int): The number of transformer layers to use.
        num_heads (int): The number of heads to use in the multi-head attention.
        attention_dropout (float): The dropout probability for attention layers.
        residual_dropout (float): The dropout probability for residual layers.
        embedding_dropout (float): The dropout probability for embedding layers.
        time_emb (bool): Whether to include time embeddings.
        use_rew (bool): Whether to include return embeddings.
        use_cost (bool): Whether to include cost embeddings.
        cost_transform (bool): Whether to transform the cost values.
        add_cost_feat (bool): Whether to add cost features.
        mul_cost_feat (bool): Whether to multiply cost features.
        cat_cost_feat (bool): Whether to concatenate cost features.
        action_head_layers (int): The number of layers in the action head.
        cost_prefix (bool): Whether to include a cost prefix.
        stochastic (bool): Whether to use stochastic actions.
        init_temperature (float): The initial temperature value for stochastic actions.
        target_entropy (float): The target entropy value for stochastic actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        time_emb: bool = True,
        use_rew: bool = False,
        use_cost: bool = False,
        cost_transform: bool = False,
        add_cost_feat: bool = False,
        mul_cost_feat: bool = False,
        cat_cost_feat: bool = False,
        action_head_layers: int = 1,
        cost_prefix: bool = False,
        stochastic: bool = False,
        init_temperature=0.1,
        target_entropy=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        if cost_transform:
            self.cost_transform = lambda x: 50 - x
        else:
            self.cost_transform = None
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat
        self.cat_cost_feat = cat_cost_feat
        self.stochastic = stochastic

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.time_emb = time_emb
        if self.time_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)

        self.seq_repeat = 2
        self.use_rew = use_rew
        self.use_cost = use_cost
        if self.use_cost:
            self.cost_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_rew:
            self.return_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1

        dt_seq_len = self.seq_repeat * seq_len

        self.cost_prefix = cost_prefix
        if self.cost_prefix:
            self.prefix_emb = nn.Linear(1, embedding_dim)
            dt_seq_len += 1

        self.blocks = nn.ModuleList([
            TransformerBlock(
                seq_len=dt_seq_len,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
            ) for _ in range(num_layers)
        ])

        action_emb_dim = 2 * embedding_dim if self.cat_cost_feat else embedding_dim

        if self.stochastic:
            if action_head_layers >= 2:
                self.action_head = nn.Sequential(
                    nn.Linear(action_emb_dim, action_emb_dim), nn.GELU(),
                    DiagGaussianActor(action_emb_dim, action_dim))
            else:
                # self.action_head = DiagGaussianActor(action_emb_dim, action_dim)
                self.act_mu = torch.nn.Linear(action_emb_dim, action_dim)
                self.act_log_std = torch.nn.Linear(action_emb_dim, action_dim)
                # act_std = torch.exp(act_log_std)
                # self.action_head = Normal(act_mu, act_std)
        else:
            self.action_head = mlp([action_emb_dim] * action_head_layers + [action_dim],
                                   activation=nn.GELU,
                                   output_activation=nn.Identity)
        # self.state_pred_head = nn.Linear(embedding_dim, state_dim)
        # self.state_pred_head = DiagGaussianActor(embedding_dim, state_dim)
        self.state_mu = torch.nn.Linear(embedding_dim, state_dim)
        self.state_log_std = torch.nn.Linear(embedding_dim, state_dim)
        # state_std = torch.exp(state_log_std)
        # self.state_pred_head = Normal(state_mu, state_std)
        # a classification problem
        self.cost_pred_head = nn.Linear(embedding_dim, 2)
        self.rtg_pred_head = nn.Linear(embedding_dim, 1)

        if self.stochastic:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

        self.apply(self._init_weights)

    def temperature(self):
        if self.stochastic:
            return self.log_temperature.exp()
        else:
            return None

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            costs_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
            episode_cost: torch.Tensor = None,  # [batch_size, ]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        seq_list = [state_emb, act_emb]

        if self.cost_transform is not None:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        if self.use_cost:
            costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, costs_emb)
        if self.use_rew:
            returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, returns_emb)

        # [batch_size, seq_len, 2-4, emb_dim], (c_0 s_0, a_0, c_1, s_1, a_1, ...)
        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len,
                                    self.embedding_dim)

        if padding_mask is not None:
            # [batch_size, seq_len * self.seq_repeat], stack mask identically to fit the sequence
            padding_mask = torch.stack([padding_mask] * self.seq_repeat,
                                       dim=1).permute(0, 2, 1).reshape(batch_size, -1)

        if self.cost_prefix:
            episode_cost = episode_cost.unsqueeze(-1).unsqueeze(-1)

            episode_cost = episode_cost.to(states.dtype)
            # [batch, 1, emb_dim]
            episode_cost_emb = self.prefix_emb(episode_cost)
            # [batch, 1+seq_len * self.seq_repeat, emb_dim]
            sequence = torch.cat([episode_cost_emb, sequence], dim=1)
            if padding_mask is not None:
                # [batch_size, 1+ seq_len * self.seq_repeat]
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # [batch_size, seq_len * self.seq_repeat, embedding_dim]
        out = self.out_norm(out)
        if self.cost_prefix:
            # [batch_size, seq_len * seq_repeat, embedding_dim]
            out = out[:, 1:]

        # [batch_size, seq_len, self.seq_repeat, embedding_dim]
        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        # [batch_size, self.seq_repeat, seq_len, embedding_dim]
        out = out.permute(0, 2, 1, 3)

        # [batch_size, seq_len, embedding_dim]
        action_feature = out[:, self.seq_repeat - 1]
        state_feat = out[:, self.seq_repeat - 2]

        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            # cost_prefix feature, deprecated
            # episode_cost_emb = episode_cost_emb.repeat_interleave(seq_len, dim=1)
            # [batch_size, seq_len, 2 * embedding_dim]
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=2)

        # get predictions

        action_mu = self.act_mu(state_feat)
        action_logstd = self.act_log_std(state_feat)
        action_std = torch.exp(action_logstd)
        self.action_distribution = Normal(action_mu, action_std)
        action_preds = self.action_distribution.rsample()
        # action_preds = self.action_head(
        #     state_feat
        # )  # predict next action given state, [batch_size, seq_len, action_dim]
        log_pi = self.action_distribution.log_prob(action_preds).sum(axis=-1)
        log_pi -= (2*(np.log(2) - action_preds - F.softplus(-2*action_preds))).sum(axis=-1)

        # [batch_size, seq_len, 2]
        cost_preds_ = self.cost_pred_head(
            action_feature)  # predict next cost return given state and action
        cost_preds_log_soft = F.log_softmax(cost_preds_, dim=-1)
        cost_preds = torch.argmax(cost_preds_log_soft).unsqueeze(0)
        # cost_preds_soft = torch.exp(cost_preds_log_soft)
        # cost_preds = torch.sum(torch.dot(cost_preds_.reshape(-1), cost_preds_soft.reshape(-1)), dim=0)
        # cost_preds = cost_preds.reshape(1, 1, -1)

        state_mu = self.state_mu(state_feat)
        state_logstd = self.state_log_std(state_feat)
        state_std = torch.exp(state_logstd)
        state_distribution = Normal(state_mu, state_std)
        state_preds = state_distribution.rsample()
        # state_preds = self.state_pred_head(
        #     action_feature)  # predict next state given state and action
        log_p = state_distribution.log_prob(state_preds).sum(axis=-1)
        log_p -= (2*(np.log(2) - state_preds - F.softplus(-2*state_preds))).sum(axis=-1)
        
        rtg_preds = self.rtg_pred_head(
            action_feature)
        rtg_preds = rtg_preds.reshape(batch_size, seq_len)
        
        log_p_pi = log_p[:, -2:-1] + log_pi[:, -1:]

        return action_preds, cost_preds, state_preds, rtg_preds, log_p_pi, cost_preds_log_soft
    

    
    def forward_with_om_no_padding(self, states, actions, returns_to_go, costs_to_go, timesteps, max_k, **kwargs):
        
        
        window_length = states.size(1)
        states_pred = []
        actions_pred = []
        rtgs_pred = []
        costs_pred = []
        log_p_pis = torch.zeros((max_k, 1), dtype=torch.float, device=states.device)

        # pad all tokens to sequence length
        for k_ in range(max_k):
            # k_len = k_ + 1
            k_len = window_length - k_

            states_ = states[:,-k_len:]
            actions_ = actions[:,-k_len:]
            returns_to_go_ = returns_to_go[:,-k_len:]
            costs_to_go_ = costs_to_go[:,-k_len:]
            timesteps_ = timesteps[:,-k_len:]
            # attention_mask_ = torch.ones(states_.shape[1]).to(dtype=torch.long, device=states.device).reshape(1, -1)

            action_pred, cost_pred, state_pred, rtg_pred, log_p_pi, _ = self.forward(
            states_, actions_, returns_to_go_, costs_to_go_, timesteps_, **kwargs)

            states_pred.append(state_pred)
            actions_pred.append(action_pred)
            rtgs_pred.append(rtg_pred)
            costs_pred.append(cost_pred)
            log_p_pis[k_] = log_p_pi
                
        gamma = 0.99
        # gamma_vector = torch.pow(gamma, torch.flip(torch.arange(max_k), dims=(0,))).cuda()
        gamma_vector = torch.pow(gamma, torch.flip(torch.arange(window_length - max_k, window_length), dims=(0,))).to(device=states.device)
        occupancy_measure = torch.tensordot(torch.exp(log_p_pi), gamma_vector, dims=([0], [0]))
        
        return states_pred[0], actions_pred[0], rtgs_pred[0].unsqueeze(-1), costs_pred[0].unsqueeze(-1).unsqueeze(-1), occupancy_measure


class CDTTrainer:
    """
    Constrained Decision Transformer Trainer
    
    Args:
        model (CDT): A CDT model to train.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        betas (Tuple[float, ...]): The betas for the optimizer.
        clip_grad (float): The clip gradient value.
        lr_warmup_steps (int): The number of warmup steps for the learning rate scheduler.
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        loss_cost_weight (float): The weight for the cost loss.
        loss_state_weight (float): The weight for the state loss.
        cost_reverse (bool): Whether to reverse the cost.
        no_entropy (bool): Whether to use entropy.
        device (str): The device to use for training (e.g. "cpu" or "cuda").

    """

    def __init__(
            self,
            model: CDT,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            betas: Tuple[float, ...] = (0.9, 0.999),
            clip_grad: float = 0.25,
            lr_warmup_steps: int = 10000,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            loss_cost_weight: float = 0.0,
            loss_state_weight: float = 0.0,
            cost_reverse: bool = False,
            no_entropy: bool = False,
            device="cpu") -> None:
        self.model = model
        self.logger = logger
        self.env = env
        self.clip_grad = clip_grad
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.cost_weight = loss_cost_weight
        self.state_weight = loss_state_weight
        self.cost_reverse = cost_reverse
        self.no_entropy = no_entropy

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )
        self.stochastic = self.model.stochastic
        if self.stochastic:
            self.log_temperature_optimizer = torch.optim.Adam(
                [self.model.log_temperature],
                lr=1e-4,
                betas=[0.9, 0.999],
            )
        self.max_action = self.model.max_action

        self.beta_dist = Beta(torch.tensor(2, dtype=torch.float, device=self.device),
                              torch.tensor(5, dtype=torch.float, device=self.device))

    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask,
                       episode_cost, costs):
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)
        action_preds, _, state_preds, rtg_preds, _, cost_preds = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            costs_to_go=costs_return,
            time_steps=time_steps,
            padding_mask=padding_mask,
            episode_cost=episode_cost,
        )

        if self.stochastic:
            log_likelihood = self.model.action_distribution.log_prob(actions)[mask > 0].mean()
            entropy = self.model.action_distribution.entropy()[mask > 0].mean()
            entropy_reg = self.model.temperature().detach()
            entropy_reg_item = entropy_reg.item()
            if self.no_entropy:
                entropy_reg = 0.0
                entropy_reg_item = 0.0
            act_loss = -(log_likelihood + entropy_reg * entropy)
            self.logger.store(tab="train",
                              nll=-log_likelihood.item(),
                              ent=entropy.item(),
                              ent_reg=entropy_reg_item)
        else:
            act_loss = F.mse_loss(self.model.action_distribution, actions.detach(), reduction="none")
            # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
            act_loss = (act_loss * mask.unsqueeze(-1)).mean()

        # cost_preds: [batch_size * seq_len, 2], costs: [batch_size * seq_len]
        cost_preds = cost_preds.reshape(-1, 2)
        costs = costs.flatten().long().detach()
        cost_loss = F.nll_loss(cost_preds, costs, reduction="none")
        # cost_loss = F.mse_loss(cost_preds, costs.detach(), reduction="none")
        cost_loss = (cost_loss * mask.flatten()).mean()
        # compute the accuracy, 0 value, 1 indice, [batch_size, seq_len]
        pred = cost_preds.data.max(dim=1)[1]
        correct = pred.eq(costs.data.view_as(pred)) * mask.flatten()
        correct = correct.sum()
        total_num = mask.sum()
        acc = correct / total_num

        # [batch_size, seq_len, state_dim]
        state_loss = F.mse_loss(state_preds[:, :-1],
                                states[:, 1:].detach(),
                                reduction="none")
        state_loss = (state_loss * mask[:, :-1].unsqueeze(-1)).mean()

        rtg_loss = F.mse_loss(rtg_preds[:, :-1],
                              returns[:, 1:].detach(),
                              reduction="none")
        rtg_loss = (rtg_loss * mask[:, :-1]).mean()

        loss = act_loss + self.cost_weight * cost_loss + self.state_weight * state_loss + 0.02 * rtg_loss

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()

        if self.stochastic:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (self.model.temperature() *
                                (entropy - self.model.target_entropy).detach())
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        self.scheduler.step()
        self.logger.store(
            tab="train",
            all_loss=loss.item(),
            act_loss=act_loss.item(),
            cost_loss=cost_loss.item(),
            cost_acc=acc.item(),
            state_loss=state_loss.item(),
            train_lr=self.scheduler.get_last_lr()[0],
        )

    def evaluate(self, num_rollouts, target_return, target_cost, prom):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(prom, self.model, self.env,
                                                      target_return, target_cost,
                                                      )
            episode_rets.append(epi_ret); episode_lens.append(epi_len); episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(episode_costs
                                    ) / self.cost_scale, np.mean(episode_lens)

    @torch.no_grad()
    def rollout(
        self,
        prom,
        model: CDT,
        env: gym.Env,
        target_return: float,
        target_cost: float,
    ) -> Tuple[float, float]:
        """
        Evaluates the performance of the model on a single episode.
        """
        states = torch.zeros(1,
                             model.episode_len + 1,
                             model.state_dim,
                             dtype=torch.float,
                             device=self.device)
        actions = torch.zeros(1,
                              model.episode_len,
                              model.action_dim,
                              dtype=torch.float,
                              device=self.device)
        returns = torch.zeros(1,
                              model.episode_len + 1,
                              dtype=torch.float,
                              device=self.device)
        costs = torch.zeros(1,
                            model.episode_len + 1,
                            dtype=torch.float,
                            device=self.device)
        time_steps = torch.arange(model.episode_len,
                                  dtype=torch.long,
                                  device=self.device)
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)
        returns[:, 0] = torch.as_tensor(target_return, device=self.device)
        costs[:, 0] = torch.as_tensor(target_cost, device=self.device)

        epi_cost = torch.tensor(np.array([target_cost]),
                                dtype=torch.float,
                                device=self.device)

        state_dim = model.state_dim
        act_dim = model.action_dim
        device = self.device
        if prom:    
            n_of_shots = 5
            len_of_prom = 5
            group_state_prom = torch.zeros((n_of_shots, len_of_prom, state_dim), device=device, dtype=torch.float32)
            group_act_prom = torch.zeros((n_of_shots, len_of_prom, act_dim), device=device, dtype=torch.float32)
            group_rtg_prom = torch.zeros((n_of_shots, len_of_prom, 1), device=device, dtype=torch.float32)
            group_cost_prom = torch.zeros((n_of_shots, len_of_prom, 1), device=device, dtype=torch.float32)

            # we keep all the histories on the device
            # note that the latest action and reward will be "padding"
            state_s0 = torch.from_numpy(obs).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            action_a0 = torch.zeros((1, act_dim), device=device, dtype=torch.float32)
            # rewards = torch.zeros(0, device=device, dtype=torch.float32)
            return_0 = torch.as_tensor(target_return).reshape(1, 1).to(device=device, dtype=torch.float32)
            cost_0 = epi_cost.reshape(1, 1).to(device=device, dtype=torch.float32)

            # ep_return = target_return
            # target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)


            # actions = torch.cat([actions[0], torch.zeros((1, act_dim), device=device)], dim=0)
            # rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            
            s = states[:, :1]
            a = actions[:, :1]
            r = returns[:, :1]
            c = costs[:, :1]
            t = time_steps[:, :1]

            act, _, _, _, _, _ = model(s, a, r, c, t, None, epi_cost)
            action_a0[0] = act

            # states_norm = (states - state_mean) / state_std

            # states_s0 = torch.unsqueeze(states_norm, 1)
            states_s0_3 = torch.unsqueeze(state_s0, 1)
            actions_a0_3 = torch.unsqueeze(action_a0, 1)


            rtg0_3 = torch.unsqueeze(return_0, 1)
            cost0_3 = torch.unsqueeze(cost_0, 1)

            input_ids1 = dict()
            input_ids1.update(states=states_s0_3)
            input_ids1.update(actions=actions_a0_3)
            input_ids1.update(returns_to_go=rtg0_3)
            input_ids1.update(costs_to_go=cost0_3)
            
            conf = {"output_mode": GenerationMode.GREEDY_SEARCH_WITH_OM_NO_PADDING}
            prom_output = generate_parallel2(model, input_ids1,
                                            generation_config=conf,
                                            num_samples=n_of_shots,
                                            max_k=5)

            # prom_output =[]

            # for _ in range(n_of_shots):
            #     input_ids1 = dict()
            #     input_ids1.update(states=states_s0_3.detach())
            #     input_ids1.update(actions=actions_a0_3.detach())
            #     input_ids1.update(returns_to_go=rtg0_3.detach())
            #     input_ids1.update(costs_to_go=cost0_3.detach())
            #     output = model.generate(input_ids1, generation_config=conf, max_k=3)
            #     prom_output.append(output)

            states_prom = [prom_out['states'][:,:-1,:][0] for prom_out in prom_output]
            actions_prom = [prom_out['actions'][:,:-1,:][0] for prom_out in prom_output]
            rtg_prom = [prom_out['returns_to_go'][:,:-1,:][0] for prom_out in prom_output]
            costs_prom = [prom_out['costs_to_go'][:,:-1,:][0] for prom_out in prom_output]
            min_om = torch.cat([torch.min(prom_out['occupancy_measure'], dim=1)[0] for prom_out in prom_output])
            argmin_om = torch.cat([torch.min(prom_out['occupancy_measure'], dim=1)[1] for prom_out in prom_output]).tolist()

            for i in range(n_of_shots):
                if argmin_om[i] + 1 < len_of_prom:
                    zero_pad_s = torch.zeros((len_of_prom - argmin_om[i] - 1, state_dim), 
                                                device=device, dtype=torch.float32)
                    zero_pad_a = torch.zeros((len_of_prom - argmin_om[i] - 1, act_dim), 
                                                device=device, dtype=torch.float32)
                    zero_pad_r = torch.zeros((len_of_prom - argmin_om[i] - 1, 1), 
                                                device=device, dtype=torch.float32)
                    group_state_prom[i] = torch.cat([zero_pad_s, states_prom[i][:argmin_om[i] + 1, :]], dim=0)
                    group_act_prom[i] = torch.cat([zero_pad_a, actions_prom[i][:argmin_om[i] + 1, :]], dim=0)
                    group_rtg_prom[i] = torch.cat([zero_pad_r, rtg_prom[i][:argmin_om[i] + 1, :]])
                else:
                    group_state_prom[i] = states_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
                    group_act_prom[i] = actions_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
                    group_rtg_prom[i] = rtg_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]

            # for i in range(n_of_shots):
            #     if argmin_om[i] + 1 < len_of_prom:
            #         group_state_prom[i] = states_prom[i][:argmin_om[i] + 1, :]
            #         group_act_prom[i] = actions_prom[i][:argmin_om[i] + 1, :]
            #         group_rtg_prom[i] = rtg_prom[i][:argmin_om[i] + 1, :]
            #         group_cost_prom[i] = costs_prom[i][:argmin_om[i] + 1, :]
            #     elif argmin_om[i] + 1 > states_prom[i].shape[0]:
            #         group_state_prom[i] = states_prom[i][argmin_om[i] - len_of_prom + 1:, :]
            #         group_act_prom[i] = actions_prom[i][argmin_om[i] - len_of_prom + 1:, :]
            #         group_rtg_prom[i] = rtg_prom[i][argmin_om[i] - len_of_prom + 1:, :]
            #         group_cost_prom[i] = costs_prom[i][argmin_om[i] - len_of_prom + 1:, :]
            #     else:
            #         group_state_prom[i] = states_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
            #         group_act_prom[i] = actions_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
            #         group_rtg_prom[i] = rtg_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]
            #         group_cost_prom[i] = costs_prom[i][argmin_om[i] - len_of_prom + 1: argmin_om[i] + 1, :]

            
            best_n = torch.argmax(min_om, dim=0).item()        
            best_state_prom = group_state_prom[best_n]
            best_act_prom = group_act_prom[best_n]
            best_rtg_prom = group_rtg_prom[best_n]
            best_cost_prom = group_cost_prom[best_n]

            # states = torch.from_numpy(obs).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            states = torch.cat([best_state_prom.reshape(1, len_of_prom, state_dim), states], dim=1)
            # actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            actions = torch.cat([best_act_prom.reshape(1, len_of_prom, act_dim), actions], dim=1)
            # rewards = torch.zeros(0, device=device, dtype=torch.float32)
            returns = torch.cat([best_rtg_prom.reshape(1, len_of_prom), returns], dim=1)
            costs = torch.cat([best_cost_prom.reshape(1, len_of_prom), costs], dim=1)


        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for step in range(model.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important) # fix this noqa!!!
            if prom:
                step = step + len_of_prom
            s = states[:, :step + 1][:, -model.seq_len:]  # noqa
            a = actions[:, :step + 1][:, -model.seq_len:]  # noqa
            r = returns[:, :step + 1][:, -model.seq_len:]  # noqa
            c = costs[:, :step + 1][:, -model.seq_len:]  # noqa
            t = time_steps[:, :step + 1][:, -model.seq_len:]  # noqa

            acts, _, _, _, _, _ = model(s, a, r, c, t, None, epi_cost)
            # if self.stochastic:
            #     acts = torch.mean(acts, dim=1)
            acts = torch.clamp(acts, -self.max_action, self.max_action)
            act = acts.reshape(-1, act_dim).cpu().numpy()[-1]
            # act = acts[0, -1].cpu().numpy()
            # act = self.get_ensemble_action(1, model, s, a, r, c, t, epi_cost)

            obs_next, reward, terminated, truncated, info = env.step(act)
            if self.cost_reverse:
                cost = (1.0 - info["cost"]) * self.cost_scale
            else:
                cost = info["cost"] * self.cost_scale
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(act)
            states[:, step + 1] = torch.as_tensor(obs_next)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
            costs[:, step + 1] = torch.as_tensor(costs[:, step] - cost)

            obs = obs_next

            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost

    def get_ensemble_action(self, size: int, model, s, a, r, c, t, epi_cost):
        # [size, seq_len, state_dim]
        s = torch.repeat_interleave(s, size, 0)
        # [size, seq_len, act_dim]
        a = torch.repeat_interleave(a, size, 0)
        # [size, seq_len]
        r = torch.repeat_interleave(r, size, 0)
        c = torch.repeat_interleave(c, size, 0)
        t = torch.repeat_interleave(t, size, 0)
        epi_cost = torch.repeat_interleave(epi_cost, size, 0)

        acts, _, _, _, _, _ = model(s, a, r, c, t, None, epi_cost)
        # if self.stochastic:
        #     acts = acts.mean

        # [size, seq_len, act_dim]
        acts = torch.mean(acts, dim=0, keepdim=True)
        # acts = acts.clamp(-self.max_action, self.max_action)
        acts = torch.clamp(acts, -self.max_action, self.max_action)
        act = acts.reshape(-1, model.action_dim).cpu().numpy()[-1]
        # act = acts[0, -1].cpu().numpy()
        return act

    def collect_random_rollouts(self, num_rollouts):
        episode_rets = []
        for _ in range(num_rollouts):
            obs, info = self.env.reset()
            episode_ret = 0.0
            for step in range(self.model.episode_len):
                act = self.env.action_space.sample()
                obs_next, reward, terminated, truncated, info = self.env.step(act)
                obs = obs_next
                episode_ret += reward
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
        return np.mean(episode_rets)
