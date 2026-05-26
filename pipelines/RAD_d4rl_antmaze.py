import os
import time 
import d4rl
import gym
import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_antmaze_dataset import MultiHorizonD4RLAntmazeDataset, D4RLAntmazeTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, dict_apply
from cleandiffuser.diffusion import ContinuousRectifiedFlow
from cleandiffuser.invdynamic import FancyMlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import set_seed
from cleandiffuser.utils.iql import IQL

from cleandiffuser.dataset.elastic_search import ElasticSearchManager1



def RetrievalOnPlan(state,dataset, horizon, k,similarity_threshold,es_manager,envtarget):
    
    
    def find_best_state(similar_states,dataset,envtarget):

        from collections import defaultdict

        grouped = defaultdict(list)
        for s in similar_states:
            grouped[s["metadata"]["path_ind"]].append(s)

        filtered_similar_states = []
        for path_ind, states in grouped.items():
            best_state = max(states, key=lambda x: x["metadata"]["discounted_return"])
            filtered_similar_states.append(best_state)

        filtered_goal_path = []
        for state in filtered_similar_states:
            path_ind = state['metadata']["path_ind"]
            a = dataset.seq_unnormal_obs[path_ind][dataset.path_lengths[path_ind]]
            path_goal = int(a[0])+int(a[1])
            if abs(envtarget-path_goal)<=2:
                filtered_goal_path.append(state)


    
        all_results = []
        state_mapping = {}
        for state in filtered_goal_path:
            path_ind = state['metadata']["path_ind"]
            timestep = state['metadata']["timestep"]
            max_timestep = dataset.path_lengths[path_ind]
            if max_timestep < timestep:
                continue

            results = []
            ts = timestep + horizon 
            if ts >= max_timestep:
                metadata = {
                'similarity':state['similarity'],
                'path_ind': path_ind,
                'timestep': max_timestep,
                'discounted_return': float(dataset.rtg[path_ind][max_timestep])
                }
            else:
                metadata = {
                    'similarity':state['similarity'],
                    'path_ind': path_ind,
                    'timestep': ts,
                    'discounted_return': float(dataset.rtg[path_ind][ts])
                }

            results.append({'metadata': metadata})
            
            if not results:
                continue
            
            all_results.extend(results)
            for result in results:
                future_timestep = result["metadata"]["timestep"]
                state_mapping[(path_ind, future_timestep)] = timestep  
        
        if not all_results:
            return None, None, None, None, None



        results_1000 = []
        results_less_1000 = []

        for result in all_results:
            path_ind = result["metadata"]["path_ind"]
            max_timestep = dataset.path_lengths[path_ind]
            
            if max_timestep == 1001:
                results_less_1000.append(result)
            else:
                results_1000.append(result)

        max_return_1000 = max((x["metadata"]["discounted_return"] for x in results_1000), default=-float('inf'))
        max_return_less_1000 = max((x["metadata"]["discounted_return"] for x in results_less_1000), default=-float('inf'))


        THRESHOLD = 0.01 
        SIGNIFICANT_DIFF = 0.08  

        best_result = None

        if max_return_1000 > -float('inf'):
    

            candidates = []
            for x in results_1000:
                if abs(x["metadata"]["discounted_return"] - max_return_1000) <= abs(max_return_1000) * THRESHOLD:
                    candidates.append(x)

            from collections import defaultdict

            grouped = defaultdict(list)
            for x in candidates:
                grouped[x['metadata']['path_ind']].append(x)

            unique_candidates = []
            for path_ind, states in grouped.items():
                best_in_path = min(states, key=lambda x: x['metadata']['timestep'])
                unique_candidates.append(best_in_path)

            best_1000 = None
            min_timestep = float('inf')
            for x in unique_candidates:
                timestep = x['metadata']['timestep']
                if timestep < min_timestep: 
                    min_timestep = timestep
                    best_1000 = x


           
            
        else:
            best_1000 = None

        if max_return_less_1000 > -float('inf'):
            candidates = []
            for x in results_less_1000:
                if abs(x["metadata"]["discounted_return"] - max_return_less_1000) < abs(max_return_less_1000 )* THRESHOLD:
                    candidates.append(x)

           
            from collections import defaultdict

            grouped = defaultdict(list)
            for x in candidates:
                grouped[x['metadata']['path_ind']].append(x)

            unique_candidates = []
            for path_ind, states in grouped.items():
                best_in_path = max(states, key=lambda x: x['metadata']['timestep'])
                unique_candidates.append(best_in_path)

            best_less_1000 = None
            max_diff = -float('inf')
            for x in unique_candidates:
                path_ind = x['metadata']['path_ind']
                timestep = x['metadata']['timestep']
                max_timestep = dataset.path_lengths[path_ind]
                diff = max_timestep - timestep
                if diff > max_diff:
                    max_diff = diff
                    best_less_1000 = x

            
        else:
            best_less_1000 = None

        if best_1000 and best_less_1000:
            if best_less_1000["metadata"]["discounted_return"] > best_1000["metadata"]["discounted_return"] * (1 + SIGNIFICANT_DIFF):
                best_less_1000_id = best_less_1000["metadata"]["path_ind"]
                best_1000_id = best_1000["metadata"]["path_ind"]
                best_less_1000_length = dataset.path_lengths[best_less_1000_id] 
                best_1000_length = dataset.path_lengths[best_1000_id] 
                
                if(best_less_1000_length > best_1000_length * (1 + SIGNIFICANT_DIFF)):
                    best_result = best_less_1000
                else:
                    best_result = best_1000
            else:
                best_result = best_1000
        elif best_1000:
            best_result = best_1000
        elif best_less_1000:
            best_result = best_less_1000

        if best_result is None:
            return None, None, None, None, None



        
        best_timestep = best_result["metadata"]["timestep"]
        best_path_ind = best_result["metadata"]["path_ind"]
        original_timestep = state_mapping.get((best_path_ind, best_timestep), None)
        best_goal_offset_1 = abs(best_timestep - original_timestep) if original_timestep is not None else None
        
        
        return original_timestep, best_result['metadata']['timestep'], best_goal_offset_1, best_result['metadata']['path_ind'],best_result['metadata']['similarity']


    similar_states = es_manager.find_similar_states_nok_antmaze_first2(
            state,
            min_score=similarity_threshold
        )

            
     
    if not similar_states:
        return None, None, None, None, None
    
    original_timestep, best_result_timestep, best_goal_offset_1, best_result_path_ind ,best_result_similarity= find_best_state(similar_states,dataset,envtarget)
    
    return   original_timestep, best_result_timestep, best_goal_offset_1, best_result_path_ind,best_result_similarity




class MultiHorizonD4RLAntmazeDatasetwQ(MultiHorizonD4RLAntmazeDataset):
    pred_values = None

    @torch.no_grad()
    def add_value(self, iql: IQL, device: str):
        self.pred_values = np.zeros_like(self.seq_rew)
        for i in tqdm(range(self.pred_values.shape[0])):
            self.pred_values[i] = iql.V(
                torch.tensor(self.seq_obs[i], device=device)).cpu().numpy()

    def __getitem__(self, idx: int):

        indices = [
            int(self.len_each_horizon[i] * (idx / self.len_each_horizon[-1])) for i in range(len(self.horizons))]

        torch_datas = []

        for i, horizon in enumerate(self.horizons):
            path_idx, start, end = self.indices[i][indices[i]]

            rewards = self.seq_rew[path_idx, start:]
            values = (rewards * self.discount[:rewards.shape[0], None]).sum(0)

            data = {
                'obs': {
                    'state': self.seq_obs[path_idx, start:end]},
                'act': self.seq_act[path_idx, start:end],
                'rew': self.seq_rew[path_idx, start:end],
                'pred_val': self.pred_values[path_idx, start:end],
                'val': values}

            torch_data = dict_apply(data, torch.tensor)

            torch_datas.append({
                "horizon": horizon,
                "data": torch_data,
            })

        return torch_datas


@hydra.main(config_path="../configs/rad/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    if args.test_model == "R2":
        args.diffusion_ckpt = args.reflow_ckpt

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    w_cfgs = [1.0, 0.0, 0.0]
    planning_horizons = [5, 5, 9]
    # ========================== Level Setup ==========================
    n_levels = len(planning_horizons)
    temporal_horizons = [planning_horizons[-1] for _ in range(n_levels)]
    for i in range(n_levels - 1):
        temporal_horizons[-2 - i] = (planning_horizons[-2 - i] - 1) * (temporal_horizons[-1 - i] - 1) + 1

    env = gym.make(args.task.env_name)
    dataset = MultiHorizonD4RLAntmazeDatasetwQ(
        env.get_dataset(), horizons=temporal_horizons, discount=args.discount)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # =========================== Model Setup ==========================
    fix_masks = [torch.zeros((h, obs_dim)) for h in planning_horizons]
    loss_weights = [torch.ones((h, obs_dim)) for h in planning_horizons]
    for i in range(n_levels):

        fix_idx = [0, -1]
        fix_masks[i][fix_idx, :] = 1.
        loss_weights[i][1, :] = args.next_obs_loss_weight

    nn_diffusions = [
        DiT1d(obs_dim, emb_dim=args.emb_dim,
              d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
        for _ in range(n_levels)]
    nn_conditions = [
        MLPCondition(1, args.emb_dim, hidden_dims=[args.emb_dim, ])
        for _ in range(n_levels)]

    diffusions = [
        ContinuousRectifiedFlow(
            nn_diffusions[i], nn_conditions[i], fix_masks[i], loss_weights[i],
            ema_rate=args.ema_rate, device=args.device)
        for i in range(n_levels)]

    invdyn = FancyMlpInvDynamic(obs_dim, act_dim, 256, nn.Tanh(), add_dropout=True, device=args.device)

    if args.mode == "iql_training":

        dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
        td_dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

        iql = IQL(obs_dim, act_dim, hidden_dim=512, discount=args.discount, tau=0.7).to(args.device)
        iql.train()

        n_gradient_step = 0
        log = dict.fromkeys(["loss_v", "loss_q"], 0.)

        log_start_time = time.time()

        for batch in loop_dataloader(td_dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            log["loss_v"] += iql.update_V(obs, act)
            log["loss_q"] += iql.update_Q(obs, act, rew, next_obs, tml)
            iql.update_target()

            if (n_gradient_step + 1) % args.log_interval == 0:
                elapsed = time.time() - log_start_time
                avg_time_per_step = elapsed / args.log_interval
                remaining_steps = args.diffusion_gradient_steps - (n_gradient_step + 1)
                eta_seconds = avg_time_per_step * remaining_steps
                eta_mins, eta_secs = divmod(eta_seconds, 60)

                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                log["ETA"] = f"{int(eta_mins)}m {int(eta_secs)}s"  # <<< 只加 ETA
                print(log)
                log = dict.fromkeys(["loss_v", "loss_q"], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                iql.save(save_path + f'iql_ckpt_{n_gradient_step + 1}.pt')
                iql.save(save_path + f'iql_ckpt_latest.pt')

            n_gradient_step += 1
            if n_gradient_step > 1_000_000:
                break

    elif args.mode == "training":

        iql = IQL(obs_dim, act_dim, hidden_dim=512).to(args.device)
        iql.load(save_path + 'iql_ckpt_latest.pt', device=args.device)
        iql.eval()

        dataset.add_value(iql, args.device)

        lr_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(diffusions[i].optimizer, args.diffusion_gradient_steps)
            for i in range(n_levels)]

        for diffusion in diffusions:
            diffusion.train()
        invdyn.train()

        n_gradient_step = 0
        log = dict.fromkeys([f"loss{i}" for i in range(n_levels)] + ["invdyn_loss"], 0.)
        disc_tensor = args.discount ** torch.arange(temporal_horizons[0], device=args.device)[None, :, None].float()
        log_start_time = time.time()
        
        for batch in loop_dataloader(dataloader):
            for i in range(n_levels):

                batch_data = batch[i]["data"]

                obs = batch_data["obs"]["state"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(
                    args.device)
                act = batch_data["act"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(args.device)
                rew = batch[i]["data"]["rew"].to(args.device)
                pred_val = batch[i]["data"]["pred_val"].to(args.device)

                rew += 1
                cum_rew = rew.cumsum(1)
                mask = (cum_rew == 0.).float()
                old_mask = mask.clone()
                mask[:, 1:] = old_mask[:, :-1]
                mask[:, 0] = 1.
                pick = (mask - old_mask).bool()
                pick[:, -1] += ~pick.any(1)
                val = rew.max(1).values / mask.sum(1)
                if i == 0:
                    val = rew - 1.
                    val[:, -1] = pred_val[:, -1]
                    val = (disc_tensor * val * mask).sum(1) / 100. + 1

                log[f"loss{i}"] += diffusions[i].update(obs, val)["loss"]
                lr_schedulers[i].step()

                if i == n_levels - 1 and n_gradient_step < args.invdyn_gradient_steps:
                    log[f"invdyn_loss"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])["loss"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                elapsed = time.time() - log_start_time
                avg_time_per_step = elapsed / args.log_interval
                remaining_steps = args.diffusion_gradient_steps - (n_gradient_step + 1)
                eta_seconds = avg_time_per_step * remaining_steps
                eta_mins, eta_secs = divmod(eta_seconds, 60)

                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                log["ETA"] = f"{int(eta_mins)}m {int(eta_secs)}s"  # <<< 只加 ETA
                print(log)
                log = dict.fromkeys([f"loss{i}" for i in range(n_levels)] + ["invdyn_loss"], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                for i in range(n_levels):
                    diffusions[i].save(save_path + f'diffusion{i}_ckpt_{n_gradient_step + 1}.pt')
                    diffusions[i].save(save_path + f'diffusion{i}_ckpt_latest.pt')
                if n_gradient_step < args.invdyn_gradient_steps:
                    invdyn.save(save_path + f'invdyn_ckpt_{n_gradient_step + 1}.pt')
                    invdyn.save(save_path + f'invdyn_ckpt_latest.pt')

            n_gradient_step += 1
            if n_gradient_step > args.diffusion_gradient_steps:
                break


    elif args.mode == "inference":

        n_candidates = args.num_candidates

        for i in range(n_levels):
            diffusions[i].load(
                save_path + f'{"reflow" if args.test_model == "R2" else "diffusion"}{i}_ckpt_{args.diffusion_ckpt}.pt')
            diffusions[i].eval()
        invdyn.load(save_path + f'invdyn_ckpt_{args.invdyn_ckpt}.pt')
        invdyn.eval()

        iql = IQL(obs_dim, act_dim, hidden_dim=512).to(args.device)
        iql.load(save_path + 'iql_ckpt_latest.pt', device=args.device)
        iql.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        priors = [torch.zeros((args.num_envs, planning_horizons[i], obs_dim),
                              device=args.device) for i in range(n_levels)]
        priors[0] = torch.zeros((args.num_envs, n_candidates, planning_horizons[0], obs_dim)).to(args.device)
        condition = torch.ones((args.num_envs, n_candidates, 1), device=args.device)
        
        es_manager = ElasticSearchManager1(
            index_name=f"{args.task.env_name}",
            state_dim=obs_dim
        )
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            count = 0

            envtarget = int(env.target_goal[0])+int(env.target_goal[1])

            while not np.all(cum_done) and t < 1000 + 1:

                goal_state = None
                b = None
                for i in range(obs.shape[0]):   
                    single_obs = obs[i]   
                    print(single_obs)

                    
                    similar_states = es_manager.find_similar_states_antmaze_first2(
                        single_obs,
                        k=1,  
                        min_score=args.minscore1,
                    )

                    if similar_states:

                       
                        original_timestep, best_result_timestep, goal_offset, best_result_path_ind,best_result_similarity = RetrievalOnPlan(
                    
                                state = single_obs,
                                dataset=dataset,
                                horizon=temporal_horizons[0],
                                k = args.k,
                                similarity_threshold=args.minscore2,
                                es_manager = es_manager,
                                envtarget = envtarget
                            )
                        
        
                        if best_result_path_ind is None :
                                
            
                                count=count+1
                        else:                       
                            b = 4
                            goal_state = dataset.seq_obs[best_result_path_ind, best_result_timestep]
                            print(best_result_path_ind,best_result_timestep)
                            print(dataset.seq_unnormal_obs[best_result_path_ind, best_result_timestep][0:1])

                          
                    else:
                        print("no_similar")
                        count=count+1



                
                target_return = np.ones(args.num_envs, dtype=np.float32)
                if "medium-play" in args.task.env_name:
                    target_return[:] = 0.2
                    target_return[obs[:, 1] > 18.] = 0.8
                elif "medium-diverse" in args.task.env_name:
                    target_return[:] = 0.2
                    target_return[obs[:, 0] > 10.] = 0.3
                    target_return[obs[:, 1] > 15.] = 0.8
                elif "large-play" in args.task.env_name:
                    target_return[:] = 0.6
                    target_return[np.logical_and(obs[:, 0] >= 13., obs[:, 1] < 28)] = 0.25
                    target_return[obs[:, 0] < 13.] = 0.1
                elif "large_diverse" in args.task.env_name:
                    target_return[:] = 0.6
                    target_return[np.logical_and(obs[:, 0] >= 13., obs[:, 1] < 28)] = 0.3
                    target_return[obs[:, 0] < 13.] = 0.25
                target_return = torch.tensor(target_return, device=args.device)[:, None, None]
                this_condition = condition * target_return

                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

        

                priors[0][:, :, 0] = obs.unsqueeze(1)

                if goal_state is not None:
                    goal_state = np.stack(goal_state, axis=0) 
                    goal_state = torch.tensor(goal_state, dtype=torch.float32, device=args.device)
                    priors[0][:, :,b] = goal_state.unsqueeze(0).unsqueeze(0).expand(1, 64, -1)


                for j in range(n_levels):
                    traj, _ = diffusions[j].sample(
                        priors[j].view(-1, planning_horizons[j], obs_dim),
                        n_samples=args.num_envs * n_candidates if j == 0 else args.num_envs,
                        sample_steps=2 if args.test_model == "R2" else 5, use_ema=args.use_ema,
                        condition_cfg=(this_condition.reshape(-1, 1) if j == 0 else this_condition[:, 0]),
                        w_cfg=w_cfgs[j], temperature=args.temperature,
                        sample_step_schedule="quad_continuous")
                    if j == 0:
                        traj = traj.reshape(args.num_envs, n_candidates, -1, obs_dim)
                        with torch.no_grad():
                            value = iql.V(traj[:, :, 1])[:, :, 0]
                            idx = torch.argmax(value, -1)
                            traj = traj[torch.arange(args.num_envs), idx]
                    if j < n_levels - 1:
                        priors[j + 1][:, [0, -1]] = traj[:, [0, 1]]

                with torch.no_grad():
                    act = invdyn(traj[:, 0], traj[:, 1]).cpu().numpy()

                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew
                print(f'[t={t}] xy: {obs[:, :2]}')
                print(f'[t={t}] rew: {ep_reward}')

            episode_rewards.append(np.clip(ep_reward, 0., 1.))

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
        print(count)

        env_eval.close()


if __name__ == "__main__":
    pipeline()
