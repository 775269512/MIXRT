import torch
import os
from network.base_net import RNN
from network.commtree_net import HistoryCommTree, StatesMixingTree
from sklearn.tree import DecisionTreeRegressor


class CommTree:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # The input dimension of RNN Tree is determined according to the parameters
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # network
        self.eval_rnn = HistoryCommTree(input_shape, args)  # A network of actions selected by each agent
        self.target_rnn = HistoryCommTree(input_shape, args)
        self.eval_commtree_net = StatesMixingTree(args)  # The network that adds up the agents Q values
        self.target_commtree_net = StatesMixingTree(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_commtree_net.cuda()
            self.target_commtree_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # Load the model if it exists
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/'+str(self.args.q_tree_depth)+'_'+str(self.args.mix_q_tree_depth)\
                        +'_b'+str(self.args.beta)+'_'+str(self.args.load_model_num)+ '_rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/'+str(self.args.q_tree_depth)+'_'+str(self.args.mix_q_tree_depth)\
                        +'_b'+str(self.args.beta)+'_'+str(self.args.load_model_num)+ '_rnn_net_params.pkl'
                path_commtree = self.model_dir + '/'+str(self.args.q_tree_depth)+'_'+str(self.args.mix_q_tree_depth)\
                        +'_b'+str(self.args.beta)+'_'+str(self.args.load_model_num)+ '_commtree_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_commtree_net.load_state_dict(torch.load(path_commtree, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_commtree))
            else:
                raise Exception("No model!")

        # Make the network parameters of target_net and eval_net the same
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_commtree_net.load_state_dict(self.eval_commtree_net.state_dict())

        self.eval_parameters = list(self.eval_commtree_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # During execution, maintain an eval_hidden for each agent
        # During the learning process, maintain an eval_hidden, target_hidden for each agent of each episode
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg CommTree')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        '''
         When learning, the extracted data is four-dimensional, and the four dimensions are 1 - the first episode 2 - the transition in the episode
         3 - The data of the first agent 4 - The specific obs dimension. Because when selecting an action, not only the current inputs need to be input, but also hidden_state input to the neural network,
         Hidden_state is related to previous experience, so it cannot be randomly selected for learning. So here multiple episodes are extracted at a time, and then given to the neural network at a time
         Pass in the transition at the same position for each episode.
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  #  Convert the data in the batch to tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()
        # Used to set the TD-error of those padded experiences to 0, so that they do not affect learning

        # Get the Q value corresponding to each agent, the dimension is
        # (number of episodes, max_episode_len, n_agents, n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()

        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # gettarget_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]


        # q_evals = torch.gather(q_evals, dim=3, index=u.unsqueeze(4).repeat(1,1,1,1,self.args.action_embed)).squeeze(3)
        #
        # q_targets[avail_u_next.unsqueeze(4).repeat(1,1,1,1,self.args.action_embed) == 0.0] = - 9999999
        # q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_commtree_net(q_evals, s)
        q_total_target = self.target_commtree_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # Erase td_error of padded experience

        # You can't use mean directly, because there is still a lot of experience that is useless,
        # sn that requires the sum and then compares the actual number of experience.o it is the real mea
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_commtree_net.load_state_dict(self.eval_commtree_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # Take out the transition_idx experience on all episodes,
        # u_onehot needs to take out all, because the previous one is used
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:
            if transition_idx == 0: # If it is the first experience, let the previous action be a 0 vector
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # Because of the current obs three-dimensional data, each dimension represents (episode number, agent number, obs dimension), directly add the corresponding vector to dim_1
            # You can, for example, add (1, 0, 0, 0, 0) to the end of agent_0, which means No. 0 among the 5 agents. And the data of agent_0 is exactly in row 0, so you need to add
            # The agent number is exactly a unit matrix, that is, the diagonal is 1, and the rest are 0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # Put together three of the obs, and put the data of episode_num episodes, self.args.n_agents agents into 40 pieces of data (40,96),
        # Because all agents here share a neural network, each piece of data has its own number, so it is still its own data.
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

            if self.args.agent_comm:
                self.eval_hidden = self.eval_hidden.reshape(-1, self.args.rnn_hidden_dim, self.args.n_agents).unsqueeze(1).repeat(1, self.args.n_agents, 1, 1)
                self.target_hidden = self.target_hidden.reshape(-1, self.args.rnn_hidden_dim, self.args.n_agents).unsqueeze(1).repeat(1, self.args.n_agents, 1, 1)

                self.eval_hidden = self.eval_hidden.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)
                self.target_hidden = self.target_hidden.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)


            # print("inputs, self.eval_hidden", inputs.shape, self.eval_hidden.shape) # torch.Size([35, 96]) torch.Size([35, 64])
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs (40,96), q_eval (40, n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # q_eval (8, 5,n_actions, -1)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # The obtained q_eval and q_target are a list, the list contains max_episode_len arrays, and the dimension of the array is (number of episodes, n_agents, n_actions)
        # Convert the list to an array of (number of episodes, max_episode_len, n_agents, n_actions)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # Initialize an eval_hidden, target_hidden for each agent in each episode
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_commtree_net.state_dict(), self.model_dir + '/'+str(self.args.q_tree_depth)+'_'+str(self.args.mix_q_tree_depth)\
                        +'_b'+str(self.args.beta)+'_'+ str(num) + '_commtree_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' +str(self.args.q_tree_depth)+'_'+str(self.args.mix_q_tree_depth)\
                        +'_b'+str(self.args.beta)+'_'+ str(num) + '_rnn_net_params.pkl')
