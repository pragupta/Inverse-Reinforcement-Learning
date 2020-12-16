"""
Run inverse reinforcement learning algorithms on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from irl.value_iteration import find_policy
from irl.value_iteration import value
from irl.value_iteration import optimal_value

def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

def main(grid_size, discount, n_objects, n_colours, n_trajectories, epochs,
         learning_rate, start_state, wind=0.0, algo="maxnet", mdp="gridworld"):
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
    trajectory_length = 8

    if mdp == "objectworld":
        import irl.mdp.objectworld as objectworld
        ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind, discount)
    elif mdp == "gridworld":
        import irl.mdp.gridworld as gridworld
        ow = gridworld.Gridworld(grid_size, wind, discount)

    ground_r  = np.array([ow.reward(s) for s in range(ow.n_states)])
    policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
                         ground_r, ow.discount, stochastic=False)
    optimal_v = optimal_value(ow.n_states, ow.n_actions,
                              ow.transition_probability,
                              normalize(ground_r), ow.discount)
    trajectories = ow.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            lambda s: policy[s],
                                            random_start=True)

    feature_matrix = ow.feature_matrix()

    print("trajectories = ", trajectories.shape)
    print("epochs = ", epochs)
    print("feature_matrix.shape = ", feature_matrix.shape)
    print("policy.shape = ", policy.shape)
#    ow.plot_grid("value_{}_t{}_e{}_w{}.png".format(algo,
#                                n_trajectories, epochs, wind), value=optimal_v)
    ow.plot_grid("policy_{}_t{}_e{}_w{}.png".format(algo,
                                n_trajectories, epochs, wind),
                                policy=policy , value=optimal_v)

    r = []
    ground_svf = []
    if algo == "maxent":
        import irl.maxent as maxent
        ground_svf = maxent.find_svf(ow.n_states, trajectories)
        r = maxent.irl(feature_matrix, ow.n_actions, discount,
                       ow.transition_probability,
                       trajectories, epochs, learning_rate)
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
                                   normalize(r), ow.discount,
                                   stochastic=False)
    recovered_v      = value(recovered_policy, ow.n_states,
                                   ow.transition_probability,
                                   normalize(r), ow.discount)

    new_trajectory = ow.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            lambda s: recovered_policy[s],
                                            True, (sx, sy))
    recovered_svf = maxent.find_svf(ow.n_states, new_trajectory)

#    ow.plot_grid("recovered_value_{}_t{}_e{}_w{}.png".format(algo,
#                                n_trajectories, epochs, wind),
#                                value=recovered_v)
    ow.plot_grid("recovered_policy_{}_t{}_e{}_w{}.png".format(algo,
                                n_trajectories, epochs, wind),
                                policy=recovered_policy,
                                value=recovered_v)




#    print("new trajectory")
#    for t in new_trajectory:
#        for s, a, rw in t:
#            print (ow.int_to_point(s), ow.actions[a], rw)
#        print ("---------")
    y, x = np.mgrid[-0.5:grid_size+0.5, -0.5:grid_size+0.5]


    plt.subplot(111)

    plt.pcolor(x, y, ground_svf.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth SVF")
    plt.savefig("ground_svf_{}_t{}_e{}_w{}.png".format(algo,
                n_trajectories, epochs, wind),
                format="png", dpi=150)

    plt.pcolor(x, y, recovered_svf.reshape((grid_size, grid_size)))
    plt.title("Recovered SVF")
    plt.savefig("recovered_svf_{}_t{}_e{}_w{}.png".format(algo,
                n_trajectories, epochs, wind),
                format="png", dpi=150)

    plt.pcolor(x, y, normalize(ground_r).reshape((grid_size, grid_size)))
    plt.title("Groundtruth reward")
    plt.savefig("ground_reward_{}_t{}_e{}_w{}.png".format(algo,
                n_trajectories, epochs, wind),
                format="png", dpi=150)

    plt.pcolor(x, y, normalize(r).reshape((grid_size, grid_size)))
    plt.title("Recovered reward")
    plt.savefig("recovered_reward_{}_t{}_e{}_w{}.png".format(algo,
                n_trajectories, epochs, wind),
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

    parser.add_argument('--wind', dest='wind',
                        default=0, type=float,
                 help='randomness in expert behavior')

    parser.add_argument('--algo', dest='algo',
                        default="maxent", type=str,
                 help='IRL algo to run')

    parser.add_argument('--mdp', dest='mdp',
                        default="gridworld", type=str,
                 help='MDP problem to solve. Currently, only support gridworld and objectworld')

    args = parser.parse_args()

    main(args.grid_size, args.discount, args.n_objects, args.n_colors,
         args.n_trajectories, args.epochs, args.lr, (args.sx, args.sy),
         args.wind, args.algo, args.mdp)

