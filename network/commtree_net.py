import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class RecurrentTreeCell(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_dim=1, tree_depth=2, comm_num=1, beta=0, cuda=False):
        super(RecurrentTreeCell, self).__init__()
        self.input_dim = input_shape
        self.hidden_shape = hidden_shape
        self.output_dim = output_dim
        self.tree_depth = tree_depth
        self.comm_num = comm_num
        self.beta = beta
        self.max_leaf_idx = None  # the leaf index with maximal path probability

        self.device = torch.device('cuda' if cuda else 'cpu')
        self._validate_parameters()

        self.q_tree_init()

    def q_tree_init(self):
        self.num_q_tree_nodes = 2 ** self.tree_depth - 1
        self.num_q_leaves = self.num_q_tree_nodes + 1  # Discretization?
        # self.hidden_shape = self.num_q_leaves

        self.q_leaves = nn.Parameter(torch.zeros([self.hidden_shape, self.num_q_leaves]),
                                     requires_grad=True)
        torch.nn.init.xavier_uniform_(self.q_leaves)

        self.transform_q_dim = nn.Linear(self.hidden_shape, self.output_dim, bias=False)
        # torch.nn.init.xavier_uniform_(self.transform_q_dim.weight)

        self.q_logit_layers = []
        for cur_depth in range(self.tree_depth):
            self.q_logit_layers.append(nn.Linear(self.input_dim, self.hidden_shape * (2 ** cur_depth)))
            torch.nn.init.xavier_uniform_(self.q_logit_layers[-1].weight)
            self.q_logit_layers[-1].bias.data.fill_(self.beta)
        self.q_logit_layers = nn.ModuleList(self.q_logit_layers)

        if self.beta:
            self.betas_q = []
            for cur_depth in range(self.tree_depth):
                beta_array = np.full((self.hidden_shape, 2 ** cur_depth), self.beta)
                self.betas_q.append(nn.Parameter(torch.FloatTensor(beta_array), requires_grad=True))
            self.betas_q = nn.ParameterList(self.betas_q)

        if not self.comm_num == 1:
            # comm with other agent or env
            self.comm_layer = []
            for h_layer in range(self.comm_num):
                h_logit_layers = []
                for cur_depth in range(self.tree_depth):
                    h_logit_layers.append(nn.Linear(self.hidden_shape, self.hidden_shape * (2 ** cur_depth)))
                    torch.nn.init.xavier_uniform_(h_logit_layers[-1].weight)
                    h_logit_layers[-1].bias.data.fill_(self.beta)
                h_logit_layers = nn.ModuleList(h_logit_layers)
                self.comm_layer.append(h_logit_layers)
        else:
            self.h_logit_layers = []
            for cur_depth in range(self.tree_depth):
                self.h_logit_layers.append(nn.Linear(self.hidden_shape, self.hidden_shape * (2 ** cur_depth)))
                torch.nn.init.xavier_uniform_(self.h_logit_layers[-1].weight)
                self.h_logit_layers[-1].bias.data.fill_(self.beta)
            self.h_logit_layers = nn.ModuleList(self.h_logit_layers)

    def forward(self, obs, hidden_state):
        """
        Forward the tree for Q of each agent.
        Return the probabilities for reaching each leaf.
        """
        batch_size = obs.size()[0]
        obs = obs.view(batch_size, -1)
        hidden_state = hidden_state.view(-1, self.comm_num, self.hidden_shape)
        device = next(self.parameters()).device

        path_prob_q = Variable(torch.ones(batch_size, self.hidden_shape, 1), requires_grad=True).to(device)
        for cur_depth in range(self.tree_depth):
            # current_prob (bs, hidden_shape, 2**cur_depth)
            current_q_logit_left = self.q_logit_layers[cur_depth](obs).view(batch_size, self.hidden_shape, -1)
            if not self.comm_num == 1:
                for h_layer in range(self.comm_num):
                    comm_state = hidden_state[:, h_layer, :]
                    current_h_logit_left = self.comm_layer[h_layer][cur_depth](comm_state) \
                        .view(batch_size, self.hidden_shape, -1)
                    current_q_logit_left = current_q_logit_left+current_h_logit_left
            else:
                current_h_logit_left = self.h_logit_layers[cur_depth](hidden_state) \
                    .view(batch_size, self.hidden_shape, -1)
                current_q_logit_left = current_q_logit_left+current_h_logit_left

            if self.beta == 0:
                current_prob_left = torch.sigmoid(current_q_logit_left).view(batch_size, self.hidden_shape,
                                                                             2 ** cur_depth)
            else:
                current_prob_left = torch.sigmoid(
                    torch.einsum("blc, lc->blc", current_q_logit_left,
                                 self.betas_q[cur_depth])).view(batch_size, self.hidden_shape, 2 ** cur_depth)

            current_prob_right = 1 - current_prob_left

            current_prob_left = current_prob_left * path_prob_q
            current_prob_right = current_prob_right * path_prob_q

            # Update next level probability
            path_prob_q = torch.stack((current_prob_left.unsqueeze(-1), current_prob_right.unsqueeze(-1)), dim=3) \
                .view(batch_size, self.hidden_shape, 2 ** (cur_depth + 1))
        _mu = torch.sum(path_prob_q, dim=1)
        vs, ids = torch.max(_mu, 1)  # ids is the leaf index with maximal path probability
        self.max_leaf_idx = ids
        # Sum the path probabilities of each layer
        # distribution_per_leaf = self.softmax(self.q_leaves)
        q_leaves = torch.einsum('bhd,hd->bh', path_prob_q, self.q_leaves)  # (bs, hidden_shape)
        q = self.transform_q_dim(q_leaves)

        return q, q_leaves

    def _validate_parameters(self):

        if not self.tree_depth > 0:
            msg = ("The Q tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.tree_depth))


class HistoryCommTree(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(HistoryCommTree, self).__init__()
        self.args = args
        if self.args.agent_comm:
            comm_num = args.n_agents
        else:
            comm_num = 1
        self.depth = args.q_tree_depth

        self.q_tree = RecurrentTreeCell(input_shape, args.rnn_hidden_dim, args.n_actions,
                                   tree_depth=args.q_tree_depth, comm_num=comm_num, beta=args.beta, cuda=args.cuda)

    def forward(self, obs, hidden_state):  # torch.Size([1, 96]) torch.Size([1, 64])
        if self.args.agent_comm:
            h_in = hidden_state.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)
        else:
            h_in = hidden_state.view(-1, self.args.rnn_hidden_dim)

        # h_in = []
        # for agent in range(self.args.n_agents):
        #     h_in.append(hidden_state[:, :, agent].view(-1, self.args.rnn_hidden_dim))

        q, h = self.q_tree(obs, h_in)
        # q = q.reshape(-1, self.args.n_actions, self.args.central_action_embed)
        return q, h

class StatesMixingTree(nn.Module):
    def __init__(self, args):
        super(StatesMixingTree, self).__init__()
        self.args = args
        self.hyper_states = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.depth = args.mix_q_tree_depth
        self.agent_tree = RecurrentTreeCell(self.args.n_agents, args.qmix_hidden_dim, self.args.n_agents,
                                       tree_depth=args.mix_q_tree_depth, comm_num=1, beta=args.beta, cuda=args.cuda)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q_values, states):  # states.shape: (episode_num, max_episode_len, state_shape)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)
        states_ = self.hyper_states(states).view(-1, self.args.qmix_hidden_dim)

        q_tot_p, _ = self.agent_tree(q_values, states_)  # The final w is produced by states with a layer tree
        q_tot_p = q_tot_p.view(-1, self.args.n_agents, 1)
        q_tot_p = self.softmax(q_tot_p).view(-1, self.args.n_agents, 1)
        q_tot = torch.bmm(q_values, q_tot_p)
        # print(q_values[0], q_tot_p[0])
        q_tot = q_tot.view(episode_num, -1, 1)  # (32, 60, 1)
        return q_tot

