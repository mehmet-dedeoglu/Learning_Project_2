import os
import numpy as np
from config import parse_args, save_module
from model2 import learning_model
from data_loader import load_batch, load_test_batch
from power import power_allocation, biconvex
from schedule import mask_generator, get_channel_coef
import gc
import sys


def main(args, save_folder):
    # Generate a learning model
    network = learning_model(args)
    loader, class_list = load_batch(args)
    test_loader = load_test_batch(args, class_list)
    # Pre-run to compute total gradient size
    batch, labels = loader[0].__next__()
    # Compute local gradient
    user_grad, grad_dim, user_initial_shape = network.grad_compute(batch, labels)

    if args.power_constant:
        power_vec = args.power_avg * np.ones(args.user_number)
    else:
        power_vec = args.power_avg*np.random.rayleigh(scale=2, size=args.user_number)
    mask_obj = mask_generator(args.subchannel_number, len(user_grad))
    remainder = np.zeros((len(user_grad), args.user_number))

    for i in range(args.iteration_number):
        # print('Iteration: ', str(i))
        if args.mask_style == "uniform":
            mask, mask_indices = mask_obj.uniform_next()
        else:
            mask, mask_indices = mask_obj.ordered_next()
        y = np.zeros((len(args.exp_style), args.subchannel_number))
        gamma = 0
        for m in range(args.user_number):
            # Load next batch of data
            batch, labels = loader[m].__next__()
            # Compute local gradient
            user_grad, user_final_shape, user_initial_shape = network.grad_compute(batch, labels)
            u_m = np.multiply(args.learning_rate, user_grad) + remainder[:, m]
            # Perform sparsification
            masked_grads = np.multiply(mask, u_m)
            # Compute remainder for next iteration
            remainder[:, m] = u_m - masked_grads
            # Sample channel fading (iid) for each sub-channel
            h_m = get_channel_coef(args.exp_style, args.subchannel_number, args.h_coef)
            # h_m = 0.000001 * np.random.rayleigh(scale=3, size=args.subchannel_number)
            # Perform power allocation
            b_km, tx_signal_m = power_allocation(args, h_m, power_vec[m], masked_grads[mask_indices])
            # Compute received signal
            y += tx_signal_m
            if "distributed" in args.exp_style:
                gamma += b_km[0][0] * h_m[0][0]
                # gamma += b_km[args.exp_style.index('distributed')][0] * h_m[args.exp_style.index('distributed')][0]
        # Perform equalization
        noise = np.random.normal(args.noise_mean, args.noise_std,
                                 (len(args.exp_style), args.subchannel_number))
        if args.exp_style[0] == 'error_free':
            rx = y
        else:
            rx = y + noise
        estimator = np.zeros((len(args.exp_style), len(user_grad)))
        for n in range(len(args.exp_style)):
            if args.exp_style[n] == 'distributed':
                estimator[n, mask_indices] = rx[n] / gamma
            elif args.exp_style[n] == 'error_free':
                estimator[n, mask_indices] = rx[n] / args.user_number
            elif args.exp_style[n] == 'single_user':
                estimator[n, mask_indices] = rx[n] / args.user_number
            elif args.exp_style[n] == 'centralized':
                print('To be implemented...')
            elif args.exp_style[n] == 'equal_power':
                estimator[n, mask_indices] = rx[n] / args.user_number
            else:
                print('Style is not defined!')
        # est_temp = y/gamma + noise/gamma
        # estimator = np.zeros(len(user_grad))
        # estimator[mask_indices] = est_temp[0]
        # Update model parameters
        network.update_params(estimator[0])
        # Compute accuracy of model at each acc iterations
        if i % args.save_interval == 0:
            batch_test, labels_test = test_loader.__next__()
            accuracy = network.check_accuracy(batch_test, labels_test, i)
            if args.exp_style[0] == 'distributed':
                save_module(args, save_folder, i, [gamma, accuracy], ['gamma', 'accuracy'])
            else:
                save_module(args, save_folder, i, [accuracy], ['accuracy'])


def central(args, save_folder):
    # Generate a learning model
    network = learning_model(args)
    loader, class_list = load_batch(args)
    test_loader = load_test_batch(args, class_list)
    # Pre-run to compute total gradient size
    batch, labels = loader[0].__next__()
    # Compute local gradient
    user_grad, grad_dim, user_initial_shape = network.grad_compute(batch, labels)

    if args.power_constant:
        power_vec = args.power_avg * np.ones(args.user_number)
    else:
        power_vec = args.power_avg * np.random.rayleigh(scale=2, size=args.user_number)
    mask_obj = mask_generator(args.subchannel_number, len(user_grad))
    remainder = np.zeros((len(user_grad), args.user_number))
    grad_save = np.zeros((len(user_grad), args.user_number))
    h_save = np.zeros((args.subchannel_number, args.user_number))

    for i in range(args.iteration_number):
        # print('Iteration: ', str(i))
        if args.mask_style == "uniform":
            mask, mask_indices = mask_obj.uniform_next()
        else:
            mask, mask_indices = mask_obj.ordered_next()
        for m in range(args.user_number):
            # Load next batch of data
            batch, labels = loader[m].__next__()
            # Compute local gradient
            user_grad, user_final_shape, user_initial_shape = network.grad_compute(batch, labels)
            u_m = np.multiply(args.learning_rate, user_grad) + remainder[:, m]
            # Perform sparsification
            masked_grads = np.multiply(mask, u_m)
            grad_save[:, m] = masked_grads
            # Compute remainder for next iteration
            remainder[:, m] = u_m - masked_grads
            # Sample channel fading (iid) for each sub-channel
            h_m = get_channel_coef(args.exp_style, args.subchannel_number, args.h_coef)
            h_save[:, m] = h_m[0]
            # h_m = 0.000001 * np.random.rayleigh(scale=3, size=args.subchannel_number)

        # Perform power allocation
        opt = biconvex(args, h_save, power_vec, grad_save[mask_indices])
        b_km, rx, alpha = opt.power_allocation_central()

        estimator = np.zeros((len(args.exp_style), len(user_grad)))
        estimator[0, mask_indices] = rx

        # Update model parameters
        network.update_params(estimator[0])
        # Compute accuracy of model at each acc iterations
        if i % args.save_interval == 0:
            batch_test, labels_test = test_loader.__next__()
            accuracy = network.check_accuracy(batch_test, labels_test, i)
            save_module(args, save_folder, i, [accuracy], ['accuracy'])


if __name__ == '__main__':
    simulation_dir = 'Simulations'
    if not os.path.exists(simulation_dir):
        os.mkdir(simulation_dir)
    argument = parse_args()
    print(argument.cuda)
    trial = 0
    sim_folder = argument.save_folder + '_' + str(trial)
    while os.path.exists(sim_folder):
        trial += 1
        sim_folder = argument.save_folder + '_' + str(trial)
    os.mkdir(sim_folder)
    setup_file = sim_folder + '/setup.txt'
    f = open(setup_file, "w+")
    f.write(str(argument))
    f.close()
    if argument.exp_style[0] == 'centralized':
        central(argument, sim_folder)
    else:
        main(argument, sim_folder)
    print('Finished')
