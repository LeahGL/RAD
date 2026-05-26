import os
import gym
import time
import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import MultiHorizonD4RLMuJoCoDataset
from cleandiffuser.dataset.elastic_search import ElasticSearchManager
from cleandiffuser.diffusion import ContinuousRectifiedFlow
from cleandiffuser.invdynamic import FancyMlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import DD_RETURN_SCALE, set_seed, loop_dataloader

def RetrievalOnPlan(state,dataset, horizon, k,similarity_threshold,es_manager):

    
    def find_best_state(similar_states):
   


        from collections import defaultdict

      
        grouped = defaultdict(list)
        for s in similar_states:
            grouped[s["metadata"]["path_ind"]].append(s)

        filtered_similar_states = []
        for path_ind, states in grouped.items():
            best_state = max(states, key=lambda x: x["metadata"]["discounted_return"])
            filtered_similar_states.append(best_state)
    
        all_results = []
        state_mapping = {}
        for state in filtered_similar_states:
            path_ind = state['metadata']["path_ind"]
            timestep = state['metadata']["timestep"]
            max_timestep = dataset.path_lengths[path_ind]
            if max_timestep < timestep:
                continue
            
            results = []
            ts = timestep + horizon 
            if ts >= max_timestep:
                break

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
                state_mapping[(path_ind, future_timestep)] = timestep  # 记录原始 state timestep
        
        if not all_results:
            return None, None, None, None, None



        results_1000 = []
        results_less_1000 = []

        for result in all_results:
            path_ind = result["metadata"]["path_ind"]
            max_timestep = dataset.path_lengths[path_ind]
            
            if max_timestep == 1000:
                results_1000.append(result)
            else:
                results_less_1000.append(result)

        max_return_1000 = max((x["metadata"]["discounted_return"] for x in results_1000), default=-float('inf'))
        max_return_less_1000 = max((x["metadata"]["discounted_return"] for x in results_less_1000), default=-float('inf'))


        THRESHOLD = 0.01  
        SIGNIFICANT_DIFF = 0.08  


        best_result = None

        if max_return_1000 > -float('inf'):
    

            candidates = []
            for x in results_1000:
                if abs(x["metadata"]["discounted_return"] - max_return_1000) < abs(max_return_1000) * THRESHOLD:
                    candidates.append(x)

            from collections import defaultdict

            grouped = defaultdict(list)
            for x in candidates:
                grouped[x['metadata']['path_ind']].append(x)


            unique_candidates = []
            for path_ind, states in grouped.items():
                best_in_path = max(states, key=lambda x: x['metadata']['timestep'])
                unique_candidates.append(best_in_path)


            best_1000 = None
            max_diff = -float('inf')
            for x in unique_candidates:
                path_ind = x['metadata']['path_ind']
                timestep = x['metadata']['timestep']
                max_timestep = dataset.path_lengths[path_ind]
                diff = max_timestep - timestep
                if diff > max_diff:
                    max_diff = diff
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


    similar_states = es_manager.find_similar_states_nok(
            state,
            min_score=similarity_threshold
        )

            
     
    if not similar_states:
        return None, None, None, None, None
    

    original_timestep, best_result_timestep, best_goal_offset_1, best_result_path_ind ,best_result_similarity= find_best_state(similar_states)
    
    return   original_timestep, best_result_timestep, best_goal_offset_1, best_result_path_ind,best_result_similarity




@hydra.main(config_path="../configs/rad/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    target_return = args.task.target_return_R2 if args.test_model == "R2" else args.task.target_return_R1
    w_cfg = args.task.w_cfg_R2 if args.test_model == "R2" else args.task.w_cfg_R1

    planning_horizons = [5, 5, 9]
    # ========================== Level Setup ==========================
    n_levels = len(planning_horizons)
    temporal_horizons = [planning_horizons[-1] for _ in range(n_levels)]
    for i in range(n_levels - 1):
        temporal_horizons[-2 - i] = (planning_horizons[-2 - i] - 1) * (temporal_horizons[-1 - i] - 1) + 1

    # ========================== Dataset and DataLoader ==========================
    env = gym.make(args.task.env_name)
    scale = DD_RETURN_SCALE[args.task.env_name]
    dataset = MultiHorizonD4RLMuJoCoDataset(
        env.get_dataset(), horizons=temporal_horizons, terminal_penalty=args.terminal_penalty, discount=args.discount)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # =========================== Model Setup ==========================
    fix_masks = [torch.zeros((h, obs_dim)) for h in planning_horizons]
    loss_weights = [torch.ones((h, obs_dim)) for h in planning_horizons]
    for i in range(n_levels):
        # fix_idx = 0 if i == 0 else [0, -1]
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

    if args.mode == "training":

        lr_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(diffusions[i].optimizer, args.diffusion_gradient_steps)
            for i in range(n_levels)]

        for diffusion in diffusions:
            diffusion.train()
        invdyn.train()

        n_gradient_step = 0
        log = dict.fromkeys([f"loss{i}" for i in range(n_levels)] + ["invdyn_loss"], 0.)

        log_start_time = time.time()

        for batch in loop_dataloader(dataloader):
            for i in range(n_levels):

                batch_data = batch[i]["data"]

                obs = batch_data["obs"]["state"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(
                    args.device)
                act = batch_data["act"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(args.device)
                val = batch_data["val"].to(args.device) / scale

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
                log_start_time = time.time()

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

        for i in range(n_levels):
            diffusions[i].load(
                save_path + f'{"reflow" if args.test_model == "R2" else "diffusion"}{i}_ckpt_{args.diffusion_ckpt}.pt')
            diffusions[i].eval()
        invdyn.load(save_path + f'invdyn_ckpt_{args.invdyn_ckpt}.pt')
        invdyn.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        priors = [torch.zeros((args.num_envs, planning_horizons[i], obs_dim),
                              device=args.device) for i in range(n_levels)]
        condition = torch.ones((args.num_envs, 1), device=args.device) * target_return
        es_manager = ElasticSearchManager(
            index_name=f"{args.task.env_name}",
            state_dim=obs_dim
        )
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            count = 0

            while not np.all(cum_done) and t < 1000 + 1:


                goal_states = []  
                for i in range(obs.shape[0]):   
                    single_obs = obs[i]   
                    similar_states = es_manager.find_similar_states(
                        single_obs,
                        k=1,  
                        min_score=args.minscore1
                    )

                    if similar_states:
                        
                        original_timestep, best_result_timestep, goal_offset, best_result_path_ind,best_result_similarity = RetrievalOnPlan(
                                state = single_obs, 
                                dataset=dataset,
                                horizon=temporal_horizons[0],
                                k =args.k,
                                similarity_threshold=args.minscore2,
                                es_manager = es_manager
                            )

                        if best_result_path_ind is None :
                                
                                goal_states.append(normalizer.normalize(single_obs))

                                count=count+1
                        else:   
                            
                            goal_state = dataset.seq_obs[best_result_path_ind, best_result_timestep]

                            goal_states.append(goal_state)

                    else:
                        print("no_similar")
                        goal_states.append(normalizer.normalize(single_obs))
                        count=count+1




                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                goal_states = np.stack(goal_states, axis=0) 
                goal_states = torch.tensor(goal_states, dtype=torch.float32, device=args.device)

                priors[0][:, 0] = obs
                priors[0][:,4] = goal_states



                for j in range(n_levels):
                    traj, _ = diffusions[j].sample(
                        priors[j],
                        n_samples=args.num_envs,
                        sample_steps=1 if args.test_model == "R2" else 3, use_ema=args.use_ema,
                        condition_cfg=condition,
                        w_cfg=w_cfg, temperature=args.temperature,
                        sample_step_schedule="quad_continuous", )
                    if j < n_levels - 1:
                        priors[j + 1][:, [0, -1]] = traj[:, [0, 1]]

                with torch.no_grad():
                    act = invdyn(traj[:, 0], traj[:, 1]).cpu().numpy()

                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
        print(count)


if __name__ == "__main__":
    pipeline()
