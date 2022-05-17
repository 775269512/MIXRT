# MIXRTs

Please adjust the parameters and run *run.sh*, or you can choose one of our fully trained models to test and visualize the structure of MIXRTs, running the following code

~~~sh
python main.py --map=3m --alg="commtree" --replay_dir="" --cuda=True --rnn_tree_dim=32 --qmix_tree_dim=16 --q_tree_depth=3 --mix_q_tree_depth=3 --load_model=True --load_model_num=6 --evaluate=True
~~~

The complete code and documentation currently under review, and will be released as soon as possible.

