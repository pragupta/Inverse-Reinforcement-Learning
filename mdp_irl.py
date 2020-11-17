"""
Run inverse reinforcement learning algorithms on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from irl.value_iteration import find_policy

def main(grid_size, discount, n_objects, n_colours, n_trajectories, epochs,
         learning_rate, start_state, algo="maxnet", mdp="gridworld"):
    """
    Run inverse reinforcement learning on the objectworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_objects: Number of objects. int.
    n_colours: Number of colours. int.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    start_state: start location to generate trajectory from
    algo: IRL algo to run (Currently, support maxnet and deep_maxnet)
    """

    sx, sy = start_state
    wind = 0.3
    trajectory_length = 8

    if mdp == "objectworld":
        import irl.mdp.objectworld as objectworld
        ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind, discount)
    elif mdp == "gridworld":
        import irl.mdp.gridworld as gridworld
        ow = gridworld.Gridworld(grid_size, wind, discount)

    ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
    policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
                         ground_r, ow.discount, stochastic=False)
    trajectories = ow.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            lambda s: policy[s])
    feature_matrix = ow.feature_matrix()

    print("trajectories = ", trajectories.shape)
    print("epochs = ", epochs)
    print("feature_matrix.shape = ", feature_matrix.shape)
    print("policy.shape = ", policy.shape)
    ow.plot_grid("policy_{}_{}_{}.png".format(algo,
                                n_trajectories, epochs), policy)

    r = []
    if algo == "maxnet":
        import irl.maxent as maxent
        r = maxent.irl(feature_matrix, ow.n_actions, discount,
                       ow.transition_probability,
                       trajectories, epochs, learning_rate, ow)
    elif algo == "deep_maxnet":
        import irl.deep_maxent as deep_maxent
        l1 = l2 = 0
        structure = (3, 3)
        r = deep_maxent.irl((feature_matrix.shape[1],) + structure,
                            feature_matrix, ow.n_actions, discount,
                            ow.transition_probability, trajectories,
                            epochs, learning_rate, l1=l1, l2=l2)

    recovered_policy = find_policy(ow.n_states, ow.n_actions,
                                   ow.transition_probability,
                                   r, ow.discount, stochastic=False)

    new_trajectory = ow.generate_trajectories(1,
                                            trajectory_length,
                                            lambda s: recovered_policy[s],
                                            False, (sx, sy))
    ow.plot_grid("recovered_policy_{}_{}_{}.png".format(algo,
                                n_trajectories, epochs), recovered_policy)
    print("new trajectory")
    for t in new_trajectory:
        for s, a, rw in t:
            print (ow.int_to_point(s), ow.actions[a], rw)
        print ("---------")
    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.savefig("reward_{}_{}_{}.png".format(algo, n_trajectories, epochs),
                format="png", dpi=150)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IRL - maxnet')
    parser.add_argument('--grid_size', dest='grid_size', type=int,
                        default=10, help='size of the grid')

    parser.add_argument('--discount', dest='discount', type=float,
                        default=0.9, help='discount factor')

    parser.add_argument('--n_objects', dest='n_objects', type=int,
                        default=15, help='Number of objects to place on the grid')

    parser.add_argument('--n_colors', dest='n_colors',
                        default=2, type=int,
                 help='Number of different colors to use for objects  on the grid')

    parser.add_argument('--n_trajectories', dest='n_trajectories',
                        default=20, type=int,
                 help='Number of trajectories to generate as input to IRL')

    parser.add_argument('--epochs', dest='epochs',
                        default=50, type=int,
                 help='Number of gradient descent steps')

    parser.add_argument('--lr', dest='lr',
                        default=0.01, type=float,
                 help='Gradient descent learning rate')

    parser.add_argument('--sx', dest='sx',
                        default=0, type=int,
                 help='x-value for the start state')

    parser.add_argument('--sy', dest='sy',
                        default=0, type=int,
                 help='x-value for the start state')

    parser.add_argument('--algo', dest='algo',
                        default="maxnet", type=str,
                 help='IRL algo to run')

    parser.add_argument('--mdp', dest='mdp',
                        default="gridworld", type=str,
                 help='MDP problem to solve. Currently, only support gridworld and objectworld')

    args = parser.parse_args()

    main(args.grid_size, args.discount, args.n_objects, args.n_colors,
         args.n_trajectories, args.epochs, args.lr, (args.sx, args.sy),
         args.algo)

