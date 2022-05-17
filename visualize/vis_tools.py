import torch
import torch.nn as nn
import sys
sys.path.append("..")
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import matplotlib as mpl
import os
import shutil


def _add_arrow(ax_parent, ax_child, xyA, xyB, color='black', linestyle=None):
    '''Private utility function for drawing arrows between two axes.'''
    con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data',
                          axesA=ax_child, axesB=ax_parent, arrowstyle='<|-',
                          color=color, linewidth=2, linestyle=linestyle)
    ax_child.add_artist(con)


def get_binary_index(depth):
    """
    Get binary index for tree nodes.
    """
    index_list = []
    for layer_idx in range(0, depth+1):
        index_list.append([bin(i)[2:].zfill(layer_idx+1) for i in range(0, np.power(2, layer_idx))])
    return np.concatenate(index_list)


def path_from_prediction(tree, idx):
    """
    Generate list of nodes as decision path,
    with each node represented by a binary string and an int index
    """
    binary_idx_list = []
    int_idx_list=[]
    idx = int(idx)
    for layer_idx in range(tree.depth+1, 0, -1):
        binary_idx_list.append(bin(idx)[2:].zfill(layer_idx))
        int_idx_list.append(2**(layer_idx-1)-1+idx)
        idx = int(idx/2)
    binary_idx_list.reverse()  # from top to bottom
    int_idx_list.reverse()
    return binary_idx_list, int_idx_list


def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def sum_nodes(leaf_nodes, num_leaf):
    #(hidden_dim, all_num_leaf)
    all_num_leaf = leaf_nodes.shape[1]
    k = int(all_num_leaf/num_leaf)
    x = torch.ones(leaf_nodes.shape[0], num_leaf)
    for i in range(num_leaf):
        x[:,i] = torch.sum(leaf_nodes[:, i*k:(i+1)*k], dim=1)
    return x


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def draw_decision_making_process(tree, args, obs, hidden, q_value, step=0, agent=0):
    input_shape = obs.shape[1]
    hidden_shape = hidden.shape[1]
    binary_indices = get_binary_index(tree.depth)
    inner_node_num = 2**tree.depth - 1
    n_leaves = inner_node_num + 1
    inner_indices = binary_indices[:inner_node_num]
    leaf_indices = binary_indices[inner_node_num:]
    inner_nodes = None
    leaf_nodes = None
    transform_weight = None
    kernels = {}
    kernels_h = {}
    cur_layer = 0
    cur_layer_h = 0
    for name, param in tree.named_parameters():
        if "q_leaves" in name:
            leaf_nodes = param
        if "transform_q_dim" in name:
            transform_weight = param
        if "weight" in name:
            if "q_logit_layers" in name:
                param = param.view(args.rnn_hidden_dim, input_shape, -1)
                kernels[cur_layer] = param
                cur_layer = cur_layer+1
            if "h_logit_layers" in name:
                param = param.view(args.rnn_hidden_dim, args.rnn_hidden_dim, -1)
                kernels_h[cur_layer_h] = param
                cur_layer_h = cur_layer_h+1

    path_prob_qs = tree.path_prob_q
    layer_imp = None
    for layer in range(len(path_prob_qs)-2, len(path_prob_qs)-1):
        weight = kernels[layer]
        path_prob_q = path_prob_qs[layer]
        for node in range(weight.shape[-1]):
            w = weight[:,:,node]    # (H, inp_shape)
            p = path_prob_q[:,:,node]   # (H, k)
            imp = p @ w
            if layer_imp==None:
                layer_imp = imp
            else:
                layer_imp += imp
    layer_imp = layer_imp.squeeze().detach().numpy()
    # print(layer_imp)
    # print(transform_weight.shape)    # (out_dim, hidden_dim)
    # print(leaf_nodes.shape)  # (hidden_dim, num_leaf)

    kkk = True
    if kkk:
        font_size = 20
        plt.rcParams.update({'font.size': font_size})
        plt.figure(figsize=(n_leaves//2, n_leaves), dpi=200)
        Grid = plt.GridSpec(len(path_prob_qs), 1, wspace=0, hspace=0.6)
        n = leaf_nodes.shape[-1]
        node_num = 0
        for layer in range(0, len(path_prob_qs)):
            print(layer, path_prob_qs[layer].shape)
            num_leaf = 2**layer
            leaf_nodes_layer = sum_nodes(leaf_nodes, num_leaf)
            print("leaf_nodes_layer",leaf_nodes_layer.shape)
            q_leaves = torch.einsum('bhd,hd->bhd', path_prob_qs[layer], leaf_nodes_layer)
            out = torch.einsum('bhd, oh->bod', q_leaves, transform_weight).detach().numpy()
            print("out", out)
            out_shape = out.shape[1]
            # pianyi =int(n/(2*num_leaf))
            ax = plt.subplot(Grid[layer,0])
            width_1 = 0
            if layer==len(path_prob_qs)-1:
                width = 0.1
            else:
                width = 0.2
            for leaf in range(out.shape[-1]):
                # pianyi = pianyi+int(n/num_leaf)-1
                data = out[0, :, leaf]
                plt.bar(np.arange(out_shape)+width_1, data, width=width,label=str(node_num))
                width_1 =width_1+width
                # plt.ylim(0, np.max(out))
                # action = np.argmax(out[0, :, leaf])
                # plt.title(str(action))
                # ax.set_xticks([])
                # ax.set_yticks([])

                node_num = node_num+1
            out = np.sum(out, axis=2)
            # plt.ylim(0, np.max(out))
            # plt.bar(np.arange(out_shape), out[0, :])
            plt.rcParams.update({'font.size': font_size})
            plt.xticks(fontsize=font_size)
            ax.set_xticks(range(out_shape))
            ax.set_yticks([])

            action = np.argmax(out[0, :])
            if layer==len(path_prob_qs)-1:
                plt.title("Leaf layer action:"+str(action), fontsize=font_size)
            else:
                plt.title("Layer:{}, action:".format(layer+1)+str(action), fontsize=font_size)
        print("q_value", q_value)
        plt.savefig('./visualize/result/steps/{}_agent{}_step{}.pdf'.format(args.map, agent, step),dpi=plt.gcf().dpi,bbox_inches='tight')

        plt.show()

    FILE_PATH = './visualize/feature_data/{}/'.format(args.map)
    if os.path.isdir(FILE_PATH):
        pass
    else:
        os.mkdir(FILE_PATH)
        os.mkdir('./visualize/obs_data/{}/'.format(args.map))

    FILE_PATH = './visualize/feature_data/{}/{}/'.format(args.map, agent)
    if os.path.isdir(FILE_PATH):
        pass
    else:
        os.mkdir(FILE_PATH)
        os.mkdir('./visualize/obs_data/{}/{}/'.format(args.map, agent))

    np.save('./visualize/feature_data/{}/{}/{}.npy'.format(args.map, agent, step), layer_imp)
    np.save('./visualize/obs_data/{}/{}/{}.npy'.format(args.map, agent, step), obs.squeeze().detach().numpy())

def draw_action_making_process(tree, args, obs, hidden,step):
    input_shape = obs.shape[1]
    q_tot = tree(obs, hidden)

    q_tot_p = tree.q_tot_p.squeeze(0).squeeze().detach().numpy()
    # print("q_tot_p",q_tot_p.shape, q_tot_p)
    # q_tot_ = torch.mm(obs, q_tot_p)
    # print("q_tot",q_tot, q_tot_)

    FILE_PATH = './visualize/feature_data/{}/'.format(args.map)
    if os.path.isdir(FILE_PATH):
        pass
    else:
        os.mkdir(FILE_PATH)

    FILE_PATH = './visualize/feature_data/{}/q_tot_w/'.format(args.map)
    if os.path.isdir(FILE_PATH):
        pass
    else:
        os.mkdir(FILE_PATH)

    np.save('./visualize/feature_data/{}/q_tot_w/{}.npy'.format(args.map,step), q_tot_p)

def draw_tree(tree, args, input_img=None):
    input_shape = args.obs_shape
    if args.last_action:
        input_shape += args.n_actions
    if args.reuse_network:
        input_shape += args.n_agents

    sns.set()
    np.random.seed(0)
    binary_indices = get_binary_index(tree.depth)
    inner_node_num = 2**tree.depth - 1
    n_leaves = inner_node_num + 1
    inner_indices = binary_indices[:inner_node_num]
    leaf_indices = binary_indices[inner_node_num:]

    inner_nodes = None
    leaf_nodes = None
    transform_weight = None
    kernels = {}
    kernels_h = {}
    cur_node = 0
    cur_node_h = 0
    for name, param in tree.named_parameters():
        print("=============Layer: ====================")
        print(name, param.shape)
        if "q_leaves" in name:
            leaf_nodes = param
            # print(leaf_nodes)
            # leaf_nodes = torch.softmax(leaf_nodes,dim=1)
            # print(torch.sum(leaf_nodes, 0))
        if "transform_q_dim" in name:
            transform_weight = param
        if "bias" in name:
            pass
        if "weight" in name:
            if "q_logit_layers" in name:
                param = param.view(args.rnn_hidden_dim, input_shape, -1)
                w_num = param.shape[-1]
                for i in range(w_num):
                    #kernels[inner_indices[cur_node]] = param[:, :, i].detach().numpy()
                    kernels[inner_indices[cur_node]] = np.sum(param[:, :, i].detach().numpy(), axis=0)
                    cur_node = cur_node + 1
            if "h_logit_layers" in name:
                param = param.view(args.rnn_hidden_dim, args.rnn_hidden_dim, -1)
                w_num = param.shape[-1]
                for i in range(w_num):
                    # kernels_h[inner_indices[cur_node_h]] = param[:, :, i].detach().numpy()
                    kernels_h[inner_indices[cur_node_h]] = np.sum(param[:, :, i].detach().numpy(), axis=0)
                    cur_node_h = cur_node_h + 1

    print(leaf_nodes.shape)  # (hidden_dim, num_leaf)
    print(transform_weight.shape)  # (out_dim, hidden_dim)
    leaf_nodes = transform_weight @ leaf_nodes
    # leaf_nodes = leaf_nodes.reshape(n_leaves, args.rnn_hidden_dim, -1)
    # leaf_nodes = torch.max(leaf_nodes, dim=1)[1]
    leaf_nodes = leaf_nodes.detach().numpy().T
    print(leaf_nodes.shape)  # (out_dim, num_leaf)
    leaves = dict([(leaf_idx, np.array([leaf_dist])) for leaf_idx, leaf_dist in zip (leaf_indices, leaf_nodes)])
    print(leaves)

    fig = plt.figure(figsize=(n_leaves, n_leaves//2), dpi=600)
    gs = GridSpec(tree.depth+1, n_leaves*4,
                  height_ratios=[1]*tree.depth+[0.5])
    # Grid Coordinate X (horizontal)
    gcx = [list(np.arange(1, 2**(i+1), 2) * (2**(tree.depth+1) // 2**(i+1)))
           for i in range(tree.depth+1)]
    gcx = list(itertools.chain.from_iterable(gcx))
    axes = {}
    path = ['0']
    imshow_args = {'origin': 'upper', 'interpolation': 'None', 'cmap': plt.get_cmap('coolwarm')}#PuBuGn
    kernel_min_val = np.min(list(kernels.values()))
    kernel_h_min_val = np.min(list(kernels_h.values()))
    kernel_max_val = np.max(list(kernels.values()))
    kernel_h_max_val = np.max(list(kernels_h.values()))
    leaf_min_val = np.min(list(leaves.values()))
    leaf_max_val = np.max(list(leaves.values()))
    font_size = 20

    # print(leaves)
    # print(set([np.argmax(leaves[k]) for k in leaves.keys()
    #                   if k.startswith('0')]))
    # for k in leaves.keys():
    #     if k.startswith('0'):
    #         print(k, leaves[k])

    # draw tree nodes q
    for pos, key in enumerate(sorted(kernels.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1, gcx[pos]*2-4:gcx[pos]*2])
        axes[key] = ax

        kernel_image = kernels[key]
        print(pos, "kernel_image", kernel_image.shape)
        print(pos, kernel_image.argsort()[::-1])


        if len(kernel_image.shape)==2: # 2D image (H, W, C)
            ax.imshow(kernel_image.squeeze(), vmin=kernel_min_val, vmax=kernel_max_val, **imshow_args)
        elif len(kernel_image.shape)==1:
            vector_image = np.ones((32, 1)) @ [kernel_image]
            ax.imshow(vector_image, vmin=kernel_min_val, vmax=kernel_max_val, **imshow_args)

        rect = plt.Rectangle((0, 0), input_shape-1, args.rnn_hidden_dim-1, fill=False, color="grey", linewidth=1)
        ax.add_patch(rect)

        ax.axis('off')
        # digits = set([np.argmax(leaves[k]) for k in leaves.keys()
        #               if k.startswith(key)])
        # title = ','.join(str(digit) for digit in digits)
        title = r"$w_o^{{{}}}$".format(pos+1)
        plt.title('{}'.format(title), fontsize=font_size//2, y=0.95)

    # draw tree nodes h_t-1
    for pos, key in enumerate(sorted(kernels_h.keys(), key=lambda x:(len(x), x))):
        # print(gcx[pos]*2, gcx[pos]*2+4)
        ax = plt.subplot(gs[len(key)-1, gcx[pos]*2:gcx[pos]*2+4])
        pos1 = axes[key].get_position()
        pos2 = ax.get_position()
        # print(pos1,pos2)
        pos2 = [pos1.x1-0.010, pos1.y0, pos1.width, pos1.height]
        ax.set_position(pos2)  # set a new position
        axes[key] = ax
        # kernel_image = np.abs(kernels[key])  # absolute value
        # kernel_image = kernel_image/np.sum(kernel_image)  # normalization

        kernel_image = kernels_h[key]
        # kernel_image = np.sum(kernel_image, axis=0)

        if len(kernel_image.shape)==2: # 2D image (H, W, C)
            ax.imshow(kernel_image.squeeze(), vmin=kernel_h_min_val, vmax=kernel_h_max_val, **imshow_args)
        elif len(kernel_image.shape)==1:
            vector_image = np.ones((32, 1)) @ [kernel_image]
            ax.imshow(vector_image, vmin=kernel_h_min_val, vmax=kernel_h_max_val, **imshow_args)
        #
        # rect = plt.Rectangle((0, 0), 0, args.rnn_hidden_dim, fill=False, color="forestgreen", linewidth=1)
        # ax.add_patch(rect)
        rect = plt.Rectangle((0, 0), args.rnn_hidden_dim-1, args.rnn_hidden_dim-1, fill=False, color="grey", linewidth=1)
        ax.add_patch(rect)

        ax.axis('off')
        # digits = set([np.argmax(leaves[k]) for k in leaves.keys()
        #               if k.startswith(key)])
        # title = ','.join(str(digit) for digit in digits)
        # plt.title('{}'.format(title), fontsize=font_size)
        title = r"$w_h^{{{}}}$".format(pos+1)
        plt.title('{}'.format(title), fontsize=font_size//2, y=0.95)

    # path, _ = path_from_prediction(tree, tree.q_tree.max_leaf_idx)

    # draw tree leaves
    for pos, key in enumerate(sorted(leaves.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1,
                            gcx[len(kernels)+pos]*2-2:gcx[len(kernels)+pos]*2+2])
        axes[key] = ax
        kernel_image = leaves[key][0]
        print(pos, kernel_image.argsort()[::-1])
        print(pos, kernel_image[kernel_image.argsort()[::-1]])

        leaf_image = np.ones((args.n_actions, 1)) @ leaves[key]
        q_w = np.sum(leaves[key])
        q_w = np.around(q_w, 4)
        # leaf_image = leaves[key].reshape(args.rnn_hidden_dim, -1)
        ax.imshow(leaf_image, vmin=leaf_min_val, vmax=leaf_max_val, **imshow_args)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plt.annotate("+", xy=(pos.x0, pos.y0), xycoords='axes fraction', fontsize=font_size//2,
        #              textcoords='offset points',ha='right')
        if pos<len(leaves.keys())-1:
            plt.title(r' ${\times}$'+r"$P^{{{}}}$+".format(pos), x=1.45, y=0, fontsize=font_size//2)
        else:
            plt.title(r'  ${\times}$'+r"$P^{{{}}}$".format(pos), x=1.35, y=0, fontsize=font_size//2)

    # add arrows indicating flow
    for pos, key in enumerate(sorted(axes.keys(), key=lambda x:(len(x), x))):
        children_keys = [k for k in axes.keys()
                         if len(k) == len(key) + 1 and k.startswith(key)]
        for child_key in children_keys:
            p_rows, p_cols = axes[key].get_images()[0].get_array().shape
            c_rows, c_cols = axes[child_key].get_images()[0].get_array().shape
            # # distinguish with green and red color
            # color = 'green' if (key in path and child_key in path) else 'red'
            # _add_arrow(axes[key], axes[child_key],
            #            (c_cols//2, 1), (p_cols//2, p_rows-1), color)
            if len(key)<tree.depth:
                p_cols=0
                c_cols=0
            elif len(key)==tree.depth:
                p_cols=0

            # distinguish with solid or dotted lines
            linestyle = None if (key in path and child_key in path) else ":"
            _add_arrow(axes[key], axes[child_key],
                       (c_cols//2, 1), (p_cols//2, p_rows-1), color='black', linestyle=linestyle)

    # draw input image with arrow indicating flow into the root node
    if input_img is not None:
        ax = plt.subplot(gs[0, 0:4])
        img_min_val = np.min(input_img)
        img_max_val = np.max(input_img)
        if len(input_img.shape) == 2:  # 2D image (H, W, C)
            ax.imshow(input_img.squeeze(), clim=(0.0, 1.0), vmin=img_min_val, vmax=img_max_val, **imshow_args)
        elif len(input_img.shape) == 1:
            vector_image = np.ones((input_img.shape[0], 1)) @ [input_img]
            ax.imshow(vector_image, vmin=img_min_val, vmax=img_max_val, **imshow_args)
        ax.axis('off')
        plt.title('input', fontsize=font_size)
        # # distinguish with green and red color
        # _add_arrow(ax, axes['0'],
        #            (1, img_rows//2), (img_cols-1, img_rows//2), 'green')

        # distinguish with solid or dotted lines
        # _add_arrow(ax, axes['0'], (1, img_rows//2), (img_cols-1, img_rows//2), 'black', None)
        norm = mpl.colors.Normalize(vmin=img_min_val, vmax=img_max_val)
        sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
        sm.set_array([])
        cbaxes = fig.add_axes([0.01, 0.7, 0.03, 0.2])  # This is the position for the colorbar
        plt.colorbar(sm, ticks=np.linspace(img_min_val, img_max_val, 3), cax=cbaxes)

    # plot color bar for kernels and leaves separately
    norm = mpl.colors.Normalize(vmin=kernel_min_val,vmax=kernel_max_val)
    sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.01, 0.7, 0.03, 0.2])  # This is the position for the colorbar
    plt.colorbar(sm, ticks=np.linspace(kernel_min_val,kernel_max_val,5), cax = cbaxes)#, label="obs_weight"
    plt.title('obs weight', fontsize=font_size//2, y=-0.3, x=1.25)

    norm = mpl.colors.Normalize(vmin=kernel_h_min_val,vmax=kernel_h_max_val)
    sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.01, 0.4, 0.03, 0.2])  # This is the position for the colorbar
    plt.colorbar(sm, ticks=np.linspace(kernel_h_min_val,kernel_h_max_val,5), cax = cbaxes)#, label="hidden_weight"
    plt.title('h weight', fontsize=font_size//2, y=-0.3, x=1)

    norm = mpl.colors.Normalize(vmin=leaf_min_val,vmax=leaf_max_val)
    sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.01, 0.1, 0.03, 0.2])  # This is the position for the colorbar, second dim is y, from bottom to top in img: 0->1
    plt.colorbar(sm, ticks=np.linspace(leaf_min_val,leaf_max_val,5), cax = cbaxes)#, label="q_leaves"
    plt.title('leaf weight', fontsize=font_size//2, y=-0.3, x=1.3)

    plt.savefig('./visualize/result/agent_{}_depth_{}.pdf'.format(args.map, args.q_tree_depth),bbox_inches='tight')
    plt.show()


def draw_q_tot_tree(tree, args, input_img=None):
    input_shape = args.n_agents

    sns.set()
    np.random.seed(0)
    binary_indices = get_binary_index(tree.depth)
    inner_node_num = 2**tree.depth - 1
    n_leaves = inner_node_num + 1
    inner_indices = binary_indices[:inner_node_num]
    leaf_indices = binary_indices[inner_node_num:]

    inner_nodes = None
    leaf_nodes = None
    transform_weight = None
    kernels = {}
    cur_node = 0
    for name, param in tree.named_parameters():
        print("=============Layer: ====================")
        print(name, param.shape)
        if "q_leaves" in name:
            leaf_nodes = param
        elif 'transform_q_dim' in name:
            transform_weight = param
        if "bias" in name:
            pass
        if "weight" in name:
            if "q_logit_layers" in name:
                param = param.view(args.qmix_hidden_dim, input_shape, -1)
                w_num = param.shape[-1]
                for i in range(w_num):
                    kernels[inner_indices[cur_node]] = np.sum(param[:, :, i].detach().numpy(), axis=0)

                    # kernels[inner_indices[cur_node]] = param[:, :, i].detach().numpy()
                    cur_node = cur_node + 1

    leaf_nodes = transform_weight @ leaf_nodes
    # print("transform_weight,",transform_weight)
    leaf_nodes = leaf_nodes.detach().numpy().T
    # leaf_nodes = torch.max(leaf_nodes, dim=1)[1]
    print(leaf_nodes.shape)  # (out_dim, num_leaf)
    leaves = dict([(leaf_idx, np.array([leaf_dist])) for leaf_idx, leaf_dist in zip (leaf_indices, leaf_nodes) ])
    # print(leaves)

    fig = plt.figure(figsize=(n_leaves, n_leaves//2), dpi=600)
    gs = GridSpec(tree.depth+1, n_leaves*2,
                  height_ratios=[1]*tree.depth+[0.5])
    # Grid Coordinate X (horizontal)
    gcx = [list(np.arange(1, 2**(i+1), 2) * (2**(tree.depth+1) // 2**(i+1)))
           for i in range(tree.depth+1)]
    gcx = list(itertools.chain.from_iterable(gcx))
    axes = {}
    path = ['0']
    imshow_args = {'origin': 'upper', 'interpolation': 'None', 'cmap': plt.get_cmap('coolwarm')}
    kernel_min_val = np.min(list(kernels.values()))
    kernel_max_val = np.max(list(kernels.values()))
    leaf_min_val = np.min(list(leaves.values()))
    leaf_max_val = np.max(list(leaves.values()))
    font_size = 10

    # draw tree nodes q
    for pos, key in enumerate(sorted(kernels.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1, gcx[pos]-2:gcx[pos]+2])
        axes[key] = ax

        kernel_image = kernels[key]

        # kernel_image = np.abs(kernels[key])  # absolute value
        # kernel_image = kernel_image/np.sum(kernel_image)  # normalization
        # kernel_image = np.sum(kernel_image, axis=0)

        if len(kernel_image.shape)==2: # 2D image (H, W, C)
            ax.imshow(kernel_image.squeeze(), vmin=kernel_min_val, vmax=kernel_max_val, **imshow_args)
        elif len(kernel_image.shape)==1:
            vector_image = np.ones((args.qmix_hidden_dim//2, 1)) @ [kernel_image]
            ax.imshow(vector_image, vmin=kernel_min_val, vmax=kernel_max_val, **imshow_args)
        ax.axis('off')
        # digits = set([np.argmax(leaves[k]) for k in leaves.keys()
        #               if k.startswith(key)])
        # title = ','.join(str(digit) for digit in digits)
        # plt.title('{}'.format(title), fontsize=font_size)
        title = r"$w_q^{{{}}}$".format(pos+1)
        plt.title('{}'.format(title), fontsize=font_size, y=1)
    # path, _ = path_from_prediction(tree, tree.q_tree.max_leaf_idx)

    tot_q = []
    # draw tree leaves
    for pos, key in enumerate(sorted(leaves.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1,
                            gcx[len(kernels)+pos]-1:gcx[len(kernels)+pos]+1])
        axes[key] = ax
        leaf_image = leaves[key]
        q_w = np.sum(leaves[key])
        tot_q.append(q_w)
        q_w = np.around(q_w, 3)
        leaf_image = np.ones((args.qmix_hidden_dim//2, 1)) @ leaf_image
        # leaf_image = leaves[key].reshape(args.qmix_hidden_dim, -1)
        ax.imshow(leaf_image, vmin=leaf_min_val, vmax=leaf_max_val, **imshow_args)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plt.title(str(q_w), y=-.5, fontsize=font_size)

        # if pos<len(leaves.keys())-1:
        #     plt.title(str(q_w)+"*P{} +".format(pos), x=0.7, y=-.5, fontsize=font_size)
        # else:
        #     plt.title(str(q_w)+"*P{}".format(pos), x=0.5, y=-.5, fontsize=font_size)

        if pos<len(leaves.keys())-1:
            plt.title(r'${\times}$'+r" $P^{{{}}}$  +".format(pos), x=3, y=0, fontsize=font_size)
        else:
            plt.title(r'${\times}$'+r" $P^{{{}}}$".format(pos), x=2.5, y=0, fontsize=font_size)
    print(tot_q)
    # add arrows indicating flow
    for pos, key in enumerate(sorted(axes.keys(), key=lambda x:(len(x), x))):
        children_keys = [k for k in axes.keys()
                         if len(k) == len(key) + 1 and k.startswith(key)]
        for child_key in children_keys:
            p_rows, p_cols = axes[key].get_images()[0].get_array().shape
            c_rows, c_cols = axes[child_key].get_images()[0].get_array().shape
            # # distinguish with green and red color
            # color = 'green' if (key in path and child_key in path) else 'red'
            # _add_arrow(axes[key], axes[child_key],
            #            (c_cols//2, 1), (p_cols//2, p_rows-1), color)

            # distinguish with solid or dotted lines
            linestyle = None if (key in path and child_key in path) else "dotted"
            _add_arrow(axes[key], axes[child_key],
                       (c_cols//2, 1), (p_cols//2, p_rows-1), color='black', linestyle=linestyle)

    # draw input image with arrow indicating flow into the root node
    if input_img is not None:
        ax = plt.subplot(gs[0, 0:4])
        img_min_val = np.min(input_img)
        img_max_val = np.max(input_img)
        if len(input_img.shape) == 2:  # 2D image (H, W, C)
            ax.imshow(input_img.squeeze(), clim=(0.0, 1.0), vmin=img_min_val, vmax=img_max_val, **imshow_args)
        elif len(input_img.shape) == 1:
            vector_image = np.ones((input_img.shape[0], 1)) @ [input_img]
            ax.imshow(vector_image, vmin=img_min_val, vmax=img_max_val, **imshow_args)
        ax.axis('off')
        plt.title('input', fontsize=font_size)
        # # distinguish with green and red color
        # _add_arrow(ax, axes['0'],
        #            (1, img_rows//2), (img_cols-1, img_rows//2), 'green')

        # distinguish with solid or dotted lines
        # _add_arrow(ax, axes['0'], (1, img_rows//2), (img_cols-1, img_rows//2), 'black', None)

        norm = mpl.colors.Normalize(vmin=img_min_val, vmax=img_max_val)
        sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
        sm.set_array([])
        cbaxes = fig.add_axes([0.01, 0.7, 0.03, 0.2])  # This is the position for the colorbar
        plt.colorbar(sm, ticks=np.linspace(img_min_val, img_max_val, 3), cax=cbaxes)

    # plot color bar for kernels and leaves separately
    norm = mpl.colors.Normalize(vmin=kernel_min_val,vmax=kernel_max_val)
    sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.01, 0.6, 0.03, 0.2])  # This is the position for the colorbar
    plt.colorbar(sm, ticks=np.linspace(kernel_min_val,kernel_max_val,5), cax = cbaxes)#, label="obs_weight"
    plt.title('q weight', fontsize=font_size, y=-0.3, x=1.2)

    norm = mpl.colors.Normalize(vmin=leaf_min_val,vmax=leaf_max_val)
    sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.01, 0.2, 0.03, 0.2])  # This is the position for the colorbar, second dim is y, from bottom to top in img: 0->1
    plt.colorbar(sm, ticks=np.linspace(leaf_min_val,leaf_max_val,5), cax = cbaxes)#, label="q_leaves"
    plt.title('leaf weight', fontsize=font_size, y=-0.3, x=1.3)
    plt.savefig('./visualize/result/mixing_{}_depth_{}.pdf'.format(args.map, args.mix_q_tree_depth),bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    pass