import numpy as np
from scipy.optimize import minimize


def power_allocation(args, chan_coef, power_limit, grad):
    b_km = []
    tx_signal = []
    for i in range(len(args.exp_style)):
        if args.exp_style[i] == "error_free":
            b_km.append(chan_coef[i])
            tx_signal.append(grad)
        elif args.exp_style[i] == "distributed":
            gamma = np.sqrt(power_limit / np.sum(grad**2 / chan_coef[i]**2))
            b_km.append(gamma / chan_coef[i])
            tx_signal.append(gamma * grad)
        elif args.exp_style[i] == "centralized":
            print('Do something')
        elif args.exp_style[i] == "single_user":
            grad[np.abs(grad) < 0.00000001] += 0.00000002
            lamb = get_lambda(grad, chan_coef[i], args.noise_std, power_limit, args.eps)
            temp = chan_coef[i] * np.abs(grad) * lamb * args.noise_std - args.noise_std ** 2
            b_km = np.sqrt(np.maximum(temp, np.zeros(len(temp)))) / (chan_coef[i] * np.abs(grad))
            tx_signal.append(b_km * chan_coef[i] * grad)
        elif args.exp_style[i] == "equal_power":
            b_km.append(np.sqrt(power_limit / np.sum(grad ** 2)))
            tx_signal.append(b_km * chan_coef[i] * grad)
        else:
            print('Error: Selected experiment style does not exist!')
    tx_signal = np.array(tx_signal)
    return b_km, tx_signal


def get_lambda(g, h, sigma, E, eps):
    lam = 0
    comp = (np.abs(g) * h * sigma * lam - sigma ** 2) / (h ** 2)
    S = np.sum(np.maximum(comp, np.zeros(len(comp))))
    delta = 50000000000000000
    flag = 0
    while np.abs(S - E) > eps:
        # print(np.abs(S - E))
        if S + eps < E:
            if flag == 2:
                delta /= 2
            lam += delta
            comp = (np.abs(g) * h * sigma * lam - sigma ** 2) / (h ** 2)
            S = np.sum(np.maximum(comp, np.zeros(len(comp))))
            flag = 1
        elif S - eps > E:
            if flag == 1:
                delta /= 2
            lam -= delta
            comp = (np.abs(g) * h * sigma * lam - sigma ** 2) / (h ** 2)
            S = np.sum(np.maximum(comp, np.zeros(len(comp))))
            flag = 2
        else:
            print('Unexpected Lambda...')
            break
    return lam


class biconvex(object):
    def __init__(self, args, h_km, power_limit, grad_vec):
        self.args = args
        self.h_km = h_km
        self.power_limit = power_limit
        self.grad_vec = grad_vec
        self.sigma = self.args.noise_std
        self.user_num = self.args.user_number
        self.b_km = 100000 / self.h_km
        sum1 = np.sum(self.grad_vec, axis=1)
        sum2 = np.sum(self.b_km * self.h_km * self.grad_vec, axis=1)
        denom = self.user_num * (self.sigma ** 2 + sum2 ** 2)
        alp_eq = sum1 * sum2 / denom
        alpha = np.maximum(alp_eq, np.zeros(len(alp_eq)))
        outer_ones = np.ones((1, args.user_number))
        self.alpha = np.transpose(np.outer(outer_ones, alpha))

        temp_row = []
        for mm in range(self.args.subchannel_number):
            arr = []
            if mm > 0:
                arr.extend([np.zeros(self.user_num) for _ in range(mm)])
            arr.extend([np.ones(self.user_num)])
            if self.args.subchannel_number - mm - 1 > 0:
                arr.extend([np.zeros(self.user_num) for _ in range(self.args.subchannel_number - mm - 1)])
            temp_row.append(np.array(arr))
        self.mul_matrix = np.transpose(np.concatenate(temp_row, axis=1))

        arrays = [np.eye(self.user_num) for _ in range(self.args.subchannel_number)]
        self.eye1 = np.concatenate(arrays, axis=0)
        self.ones = np.ones(self.args.subchannel_number)
        # self.alpha = np.stack((in_arr1, in_arr2), axis = 0)

    def power_allocation_central(self):
        for i in range(100):
            # Update b_km
            # print(np.sum(self.alpha * (self.h_km * self.b_km * self.grad_vec), axis=1))
            # print(np.sum(self.grad_vec, axis=1)/2)
            self.update_b()
            print('Optimization step: ' + str(i))
            # Update alpha
            sum1 = np.sum(self.grad_vec, axis=1)
            sum2 = np.sum(self.b_km * self.h_km * self.grad_vec, axis=1)
            denom = self.user_num * (self.sigma ** 2 + sum2 ** 2)
            alp_eq = sum1 * sum2 / denom
            alpha = np.maximum(alp_eq, np.zeros(len(alp_eq)))
            outer_ones = np.ones((1, self.user_num))
            self.alpha = np.transpose(np.outer(outer_ones, alpha))

        noise = np.random.normal(self.args.noise_mean, self.args.noise_std,
                                 (len(self.args.exp_style), self.args.subchannel_number))
        tx_signal = alpha * np.sum((self.h_km * self.b_km * self.grad_vec), axis=1) + noise
        return self.b_km, tx_signal, alpha

    def update_b(self):
        shape_b = self.b_km.shape
        shape_h = self.h_km.shape
        shape_alpha = self.alpha.shape
        shape_grad = self.grad_vec.shape

        self.b_km = np.reshape(self.b_km, (self.user_num * self.args.subchannel_number,))
        self.h_km = np.reshape(self.h_km, (self.user_num * self.args.subchannel_number,))
        self.alpha = np.reshape(self.alpha, (self.user_num * self.args.subchannel_number,))
        self.grad_vec = np.reshape(self.grad_vec, (self.user_num * self.args.subchannel_number,))

        b1 = self.b_km
        bb = (0, 100000000000000000000)
        bnds = [bb for _ in range(len(b1))]
        con_def = {'type': 'ineq', 'fun': self.cons_fun}
        cons = [con_def]
        # aaa1 = self.objective(b1)
        # bbb1 = self.cons_fun(b1)
        sol = minimize(self.objective, b1, method='COBYLA', bounds=bnds, constraints=cons)
        aaa2 = self.objective(sol.x)
        bbb2 = self.cons_fun(sol.x)
        print('Objective after: ' + str(aaa2))
        print('Constraint after: ' + str(bbb2))
        if min(bbb2) < 0:
            sol.x /= 1.5
        self.b_km = sol.x

        self.b_km = np.reshape(self.b_km, shape_b)
        self.h_km = np.reshape(self.h_km, shape_h)
        self.alpha = np.reshape(self.alpha, shape_alpha)
        self.grad_vec = np.reshape(self.grad_vec, shape_grad)

    def objective(self, x):
        cost = np.dot(np.dot((self.alpha * x * self.h_km - 1/self.user_num) * self.grad_vec,
                             self.mul_matrix) ** 2, self.ones) * 1000000000000000
        return cost

    def cons_fun(self, x):
        cost1 = self.power_limit - np.dot((x * self.grad_vec) ** 2, self.eye1)
        return cost1
