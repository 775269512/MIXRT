import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import seaborn as sns
import matplotlib.pyplot as plt
from visualize.vis_tools import draw_tree, draw_q_tot_tree

class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # Used to save PLT and pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.args.evaluate and self.args.load_model and self.args.is_tree:
            draw_tree(self.agents.policy.eval_rnn, self.args)
            draw_q_tot_tree(self.agents.policy.eval_commtree_net, self.args)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
                if episode_reward<0.05 and time_steps>self.args.evaluate_cycle*2:
                    return num
            print('Run {}, time_steps {}'.format(num, time_steps))
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, win_tag, steps = self.rolloutWorker.\
                    generate_episode(episode_idx)
                print("train train train", episode['r'].shape, episode_reward, win_tag, steps)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)
        return num+1

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            episode, episode_reward, win_tag, steps = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            print("evaluate evaluate evaluate",episode['r'].shape, episode_reward, win_tag, steps)

            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        if not self.args.is_tree:
            plt.savefig(self.save_path + '/plt'+'_{}.png'.format(num), format='png')
            np.save(self.save_path + '/win_rates'+'_{}'.format(num), self.win_rates)
            np.save(self.save_path + '/episode_rewards'+'_{}'.format(num), self.episode_rewards)
        else:
            plt.savefig(self.save_path + '/plt_'+str(self.args.q_tree_depth)+'_'+str(self.args.mix_q_tree_depth)\
                        +'_b'+str(self.args.beta)+'_{}_{}_{}.png'.format(self.args.rnn_hidden_dim, self.args.qmix_hidden_dim, num), format='png')
            np.save(self.save_path + '/win_rates_'+str(self.args.q_tree_depth)+'_'+str(self.args.mix_q_tree_depth)\
                        +'_b'+str(self.args.beta)+'_{}_{}_{}'.format(self.args.rnn_hidden_dim, self.args.qmix_hidden_dim, num), self.win_rates)
            np.save(self.save_path + '/episode_rewards_'+str(self.args.q_tree_depth)+'_'+str(self.args.mix_q_tree_depth)\
                        +'_b'+str(self.args.beta)+'_{}_{}_{}'.format(self.args.rnn_hidden_dim, self.args.qmix_hidden_dim, num), self.episode_rewards)
        plt.close()









