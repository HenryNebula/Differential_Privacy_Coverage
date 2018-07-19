from __future__ import division
from scipy.special import comb
from scipy.stats import norm
from scipy.optimize import fmin_l_bfgs_b as fmin
import os
import random
from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from sklearn.linear_model import LinearRegression
import json
from read_data import *
import math
import copy
from util import *

def Gaussian_Approx(X, mu, std):
    if std == 0:
        return 0
    else:
        return -norm.cdf(x=(X-0.5-mu)/std) + norm.cdf(x=(X+0.5-mu)/std)

def posterior_core(X, plc_id, p_0, p_1, N, Xi, comb_mat):
        xi = Xi[plc_id]
        upper_bound = 1000
        sum_ = 0
        # use Gaussian to approximate Binomial when N is large
        if N > upper_bound:
            mu = N * p_0
            std = math.sqrt(N * p_0 * (1 - p_0))
            numerator = Gaussian_Approx(X, mu, std) * (1 - xi) ** N
        else:
            numerator = p_0 ** X * (1 - p_0) ** (N - X) * comb_mat[N, X] * (1 - xi) ** N
        for i in range(0, N + 1):
            if N > upper_bound:
                mu = N * xi
                std = math.sqrt(N * xi * (1 - xi))
                outer = Gaussian_Approx(i, mu, std)
            else:
                outer = comb_mat[N, i] * xi ** i * (1 - xi) ** (N - i)
            if outer == 0.0:
                continue
            inner = 0
            for m in range(max([0, X + i - N]), min([i, X]) + 1):
                if i <= upper_bound:
                    first_part = comb_mat[i, m] * p_1 ** m * (1 - p_1) ** (i - m)
                else:
                    mu = i * (1 - p_1)
                    std = math.sqrt(i * p_1 * (1 - p_1))
                    first_part = Gaussian_Approx(m, mu, std)
                if N - i > upper_bound:
                    mu = (N - i) * p_0
                    std = math.sqrt((N - i) * p_0 * (1 - p_0))
                    second_part = Gaussian_Approx(X - m, mu, std)
                else:
                    second_part = comb_mat[N - i, X - m] * p_0 ** (X - m) * (1 - p_0) ** (N - i - X + m)
                inner += first_part * second_part
            sum_ += outer * inner
        ratio = numerator / sum_
        return ratio


class Diff_Coverage():

    def __init__(self, uniform=True, f=0, flip_p=0.001, flip_q = 0.1, data_src="SG",
                 plc_num=200,people=500, candidate_num=50, k_favor=5,
                 max_iter=200, stop_iter=200, freeze=False, skew=False, random_start=5,
                 new_set=False, obfuscate=0):
        # constants in rappor
        self.f = f
        self.p = flip_p
        self.q = flip_q
        self.eps = 4
        # constants in datasets
        self.plc_num = plc_num
        self.k_favor = k_favor
        self.people = people
        self.candidate_num = candidate_num

        # vector, different xi for different place if not uniform
        self.uniform = uniform
        self.xi = None
        self.data_source = data_src

        # constants in iteration
        self.comb_mat = np.load('comb_mat.npy')
        self.iter = max_iter
        # used in greedy random select
        self.stop_iter = stop_iter

        # save to disk
        self.sample_dir = "sample.npy"
        self.transfer_dir = "transfer.npy"

        # paras for regression: a for slope, b for intersect
        # vector, different
        self.a = None
        self.b = None
        self.prior_dict = {}

        self._is_train = False
        self.transferred_sample = None
        self.true_sample = None

        self.freeze = freeze
        self.new_set = new_set
        self.prob_mat = None
        self.skew = skew
        self.random_start = random_start
        self.obfuscate = obfuscate

    # make table for combination numbers
    def make_table(self):
        comb_mat = np.zeros((self.candidate_num + 1, self.candidate_num + 1))
        for i in range(0, self.candidate_num + 1):
            for j in range(0, i + 1):
                comb_mat[i][j] = comb(i, j)
        np.save('comb_mat', comb_mat)


    def update_paras(self):
        self.a = np.zeros(self.plc_num)
        self.b = np.zeros(self.plc_num)

    def update_prior(self):
        if self.uniform:
            self.xi = np.ones(self.plc_num) * self.k_favor / (self.plc_num + 0.0)
            self.prior_dict[self.xi[0]] = 0
        else:
            self.xi = np.sum(self.true_sample, axis=0) / self.people

        self.xi = self.xi + np.random.rand(self.xi.shape[0]) * self.obfuscate
        self.xi[self.xi < 0] = 0.0
        self.xi[self.xi > 1] = 1

    def posterior(self, X, plc_id):
        N = self.candidate_num
        p_0 = self.p - 0.5 * self.f * (self.p - self.q)
        p_1 = self.q + 0.5 * self.f * (self.p - self.q)
        Xi = self.xi
        comb_mat = self.comb_mat
        return posterior_core(X, plc_id, p_0, p_1, N, Xi, comb_mat)

    def optimizer(self, init_, metric='osine'):
        def optim_posterior(init_, metric):
            p= init_[0]
            q = 1.0/((1.0-p)/(p*np.exp(self.eps)) + 1)
            p_0 = p
            p_1 = q
            N = self.people
            Ratio_dict = {}
            count = 0
            Ratio = np.zeros((self.plc_num))
            for plc_id in range(self.plc_num):
                xi = self.xi[plc_id]
                if Ratio_dict.has_key(xi):
                    Ratio[plc_id] = Ratio_dict[xi]
                else:
                    T = int(np.around(N * xi))
                    X = int(np.around(T * q + (N - T) * p))
                    Xi = self.xi
                    comb_mat = self.comb_mat
                    ratio = posterior_core(X, plc_id, p_0, p_1, N, Xi, comb_mat)
                    Ratio[plc_id] = ratio
                    Ratio_dict[xi] = ratio
                    count += 1

            # print "Total different prior: ", count
            if metric == "RMSE":
                loss = np.sqrt(np.average(norm(Ratio - self.xi)**2))
            elif metric == 'Cosine':
                post = 1 - Ratio
                loss = np.dot(post, self.xi)/norm(post)/norm(self.xi)
                loss = 0.5 + 0.5*loss
            else:
                loss = np.sum(Ratio)
            print loss, p
            return loss
        init_ = np.array(init_)
        optim_posterior(init_, metric)

    # read real dataset from different resources
    def real_sample(self, draw=False):
        sample = []
        if self.data_source == "MCS" or "SG" or "CG":
            if self.data_source == "MCS":
                rd = MCS(k_favor=self.k_favor, people=self.people)
                sample = rd.make_grid()
                self.plc_num = rd.plc_num
            elif self.data_source == "SG":
                rd = SG(k_favor=self.k_favor, people=self.people, max_id=self.plc_num)
                sample = rd.make_grid()
            else:
                rd = CG(k_favor=self.k_favor, people=self.people, max_id=self.plc_num)
                sample = rd.make_grid()
            if self.skew:
                print "Not real dataset, add artificial groups"
                num = int(self.plc_num/self.k_favor)
                full_group = np.zeros((num, self.plc_num))
                for row in range(0,num):
                    full_group[row, range((row-1)*self.k_favor, row*self.k_favor)] = 1
                sample = np.vstack((sample, full_group))
                self.people = sample.shape[0]
            np.save(self.sample_dir, sample)
            print "Total recruit: ", self.people, " from places: ", self.plc_num
            self.update_paras()
        else:
            print "Choice is wrong"
            return

        row_sum = np.sum(sample, axis=0)
        if draw:
            plt.plot(np.arange(0, self.plc_num), row_sum)
            plt.show()
        return sample

    def flip(self, bit):
        r = np.random.rand()
        if r < 0.5 * self.f:
            B_prime = 1
        elif r < self.f:
            B_prime = 0
        else:
            B_prime = bit
        r = np.random.rand()
        if B_prime == 1:
            if r < self.q:
                B = 1
            else:
                B = 0
        else:
            if r < self.p:
                B = 1
            else:
                B = 0
        return B

    # create "fake" dataset after random response
    def transfer_sample(self, draw=False):
        if self.new_set:
            sample = self.real_sample()
        else:
            sample = np.load(self.sample_dir)
        self.update_paras()
        self.true_sample = sample.copy()
        self.update_prior()

        for i in range(0, self.people):
            r = sample[i, :]
            r = np.array(map(lambda bit: self.flip(bit), r))[np.newaxis, :]
            sample[i, :] = r

        row_sum = np.sum(sample, axis=0)
        if draw:
            plt.plot(np.arange(0, self.plc_num), row_sum)
            plt.show()
        self.transferred_sample = sample
        np.save(self.transfer_dir, sample)

    # sum posterior in terms of locations
    def sum_posterior(self, sum_row):
        if self.uniform and self.prob_mat is not None:
            posterior_per_loc = [self.prob_mat[int(x)] for x in sum_row]
        else:
            posterior_per_loc = [self.posterior(int(x), id) for id, x in enumerate(sum_row)]
        total_sum = np.sum(np.array(posterior_per_loc))
        return total_sum

    def train(self, use_grad=False, fake=False):
        if not os.path.exists(self.sample_dir) or \
                not os.path.exists(self.transfer_dir) or not self.freeze:
            self.transfer_sample()
            sample = self.transferred_sample
        else:
            self.transferred_sample = np.load(self.transfer_dir)
            self.true_sample = np.load(self.sample_dir)
            sample = self.transferred_sample
            if not self.uniform and not fake:
                self.update_paras()
                self.update_prior()

        # first fit the linear regression coefficients
        self.posterior_regression(draw=False)

        min_loss = 1e10
        final_choice = []
        for times in range(0, self.random_start):
            # choice means those ids which are in the candidate set
            choice = random.sample(range(self.people), self.candidate_num)
            not_choice = list(set(range(self.people)).difference(set(choice)))

            # party means the bit arrays for those candidates
            party = sample[choice, :]
            unused = sample[not_choice, :]

            sum_row = np.sum(party, axis=0)
            total_sum = self.sum_posterior(sum_row)
            # print "Start Training: Original Loss - " + str(total_sum)

            same_count = 0
            past_set = set()
            for i in range(0, self.iter):
                if use_grad:
                    diff, id_of_r, id_of_r_ = self.grad_boosting(
                        sum_row, party, unused)
                    r = choice[id_of_r]
                    r_ = not_choice[id_of_r_]
                    if diff > 0:
                        if {r,r_} == past_set:
                            same_count += 1
                        else:
                            past_set = {r, r_}
                            same_count = 0
                        if same_count >= 3:
                            print "Stop iteration at: ", i
                            break
                        choice[id_of_r] = r_
                        not_choice[id_of_r_] = r
                        party = sample[choice, :]
                        unused = sample[not_choice, :]
                        sum_row = np.sum(party, axis=0)

                        # print 'Loss reduce by: ', str(diff), 'r and r_ is: ', r, r_
                    else:
                        break
                else:
                    choice = self.intelli_grad()
            total_sum = self.sum_posterior(sum_row)
            print 'Finish:', total_sum, same_count, "Random start: ", times
            if total_sum < min_loss:
                min_loss = total_sum
                final_choice = choice
        # final_choice = self.intelli_grad()
        self._is_train = True
        return final_choice

    def perfect_baseline(self, true_sample):

        plcs = set()
        final_choice = []
        count = 0
        mask = np.ones(true_sample.shape)
        old_sum = 0
        while count <= self.candidate_num:
            true_sample = true_sample*mask
            person_contrib = np.sum(true_sample, 1)
            id_ = np.argmax(person_contrib)
            new_plcs = np.where(true_sample[int(id_), :] == 1)[0]
            [plcs.add(p) for p in new_plcs]
            mask[:, np.array(new_plcs, dtype=np.int64)] = 0
            count = count + 1
            if old_sum == len(plcs):
                print 'only needs: ', count
                whole_group = set(range(true_sample.shape[0]))
                newly_add = whole_group.difference(set(final_choice))
                final_choice += list(newly_add)[0:self.candidate_num - len(final_choice)]
                break
            else:
                final_choice.append(id_)
                old_sum = len(plcs)
        max_sum = len(plcs)
        return max_sum, final_choice

    # compare with two baseline: use "fake data" directly and greedy random search
    def validate(self, choice, times=1):
        if not self._is_train:
            exit(1)
        print 'Non-zero prior: ', len(np.where(self.xi > 0)[0])
        true_sample = self.true_sample
        fake_sample = self.transferred_sample

        result = []
        print 'Perfect Baseline'
        max_sum, final_choice = self.perfect_baseline(true_sample)
        result.append((max_sum, len(np.where(self.xi > 0)[0])))

        true_opt_party = true_sample[choice, :]
        fake_opt_party = fake_sample[choice, :]
        sum_row = np.sum(fake_opt_party, axis=0)
        total_sum_opt = self.sum_posterior(sum_row)
        coverage_opt = np.sum(np.sum(true_opt_party, axis=0) > 0)
        result.append((coverage_opt, total_sum_opt))
        for i in range(times):
            max_sum, final_choice = self.perfect_baseline(fake_sample)
            final_party = true_sample[final_choice, :]
            real_sum = np.sum(np.sum(final_party, axis=0) > 0)
            result.append((real_sum, max_sum))

            rand_choice = random.sample(range(self.people), self.candidate_num)
            true_rand_party = true_sample[rand_choice, :]
            fake_rand_party = fake_sample[rand_choice, :]

            # print out the posterior
            sum_row = np.sum(fake_rand_party, axis=0)
            total_sum_rand = self.sum_posterior(sum_row)
            coverage_rand = np.sum(np.sum(true_rand_party, axis=0) > 0)
            result.append((coverage_rand, total_sum_rand))

        # first element is opt, second is random-greedy, third is fake-data
        return result

    def posterior_regression(self, draw=False):
        # now coefficients should be a matrix, each place has its own slope and intercept

        # only take several points to fit the linear regression,
        # avoiding high cost of iteration

        # x_max = self.candidate_num + 1
        x_max = 10
        def find_coeffs(plc_id):
            X_range = np.arange(0, x_max)
            stop_count = self.candidate_num + 1
            ans = np.zeros(X_range.shape)
            for X in X_range:
                ans[X] = np.log(self.posterior(X=X, plc_id=plc_id))
                if np.isinf(ans[X]) or np.isnan(ans[X]):
                    ans[X] = 0
                    stop_count = X
                    break
            X_range = X_range[0:stop_count, np.newaxis]
            ans = ans[0:stop_count, np.newaxis]
            if (draw):
                plt.scatter(X_range, ans)
                plt.show()
            if ans.shape[0] == 1:
                coef = 0.0, 0.0
            else:
                lr = LinearRegression()
                lr.fit(X_range, ans)
                coef = lr.coef_, lr.intercept_
            return coef

        if self.uniform:
            # print "Make posterior tables!"
            self.prob_mat = np.array([self.posterior(x,1) for x in range(self.candidate_num + 1)])
            self.prob_mat[np.isnan(self.prob_mat)] = 0
            coeffs = find_coeffs(plc_id=0)
            self.a = np.ones(self.a.shape) * coeffs[0]
            self.b = np.ones(self.b.shape) * coeffs[1]
        else:
            # self.prob_mat = np.zeros((self.plc_num, self.candidate_num + 1))
            # plc with same prior just need one time calculation
            prior_dict = {}

            for plc_id in range(self.plc_num):
                xi_id = int(np.around(self.xi[plc_id] * self.people))
                if prior_dict.has_key(xi_id):
                    # self.prob_mat[plc_id,:] = self.prob_mat[prior_dict[xi_id],:]
                    self.a[plc_id] = self.a[prior_dict[xi_id]]
                    self.b[plc_id] = self.b[prior_dict[xi_id]]
                else:
                    # for x in range(0,self.candidate_num+1):
                    #     self.prob_mat[plc_id, x] = self.posterior(X=x, plc_id=plc_id)
                    prior_dict[xi_id] = plc_id
                    coeffs = find_coeffs(plc_id)
                    self.a[plc_id] = coeffs[0]
                    self.b[plc_id] = coeffs[1]
                # if xi_id == 0:
                #     print self.a[plc_id], self.b[plc_id]
            print "Total different prior: ", len(prior_dict)

        print "Linear Regression over"

    def grad_boosting(self, sum_row, party, unused):
        grad_ = abs(self.a * np.exp(self.b) * np.exp(self.a * sum_row)).T
        candidate_sub = np.dot(party, grad_)
        id_of_r = np.argmin(candidate_sub)

        # re-calculate grad_ using party after removing a candidate
        party[id_of_r,:] = 0
        sum_row = np.sum(party, axis=0)
        grad_ = abs(self.a * np.exp(self.b) * np.exp(self.a * sum_row)).T
        candidate_add = np.dot(unused, grad_)

        # return their sequential id, not original id in sample
        id_of_r_ = np.argmax(candidate_add)

        diff = np.max(candidate_add) - np.min(candidate_sub)
        return diff, int(id_of_r), int(id_of_r_)

    def intelli_grad(self):
        sum_row = 0
        count = 0
        temp_sample = self.transferred_sample
        choice = []
        while count <= self.candidate_num:
            # grad_ = abs(self.a * np.exp(self.b) * np.exp(self.a * sum_row)).T
            grad_ = abs((np.exp(self.a) - 1) * np.exp(self.a * sum_row + self.b)).T
            candidate_add = np.dot(temp_sample, grad_)
            id_of_r_ = np.argmax(candidate_add)
            max_val = np.max(candidate_add)
            if max_val != 0:
                choice.append(id_of_r_)
                sum_row += temp_sample[int(id_of_r_), :]
                temp_sample[int(id_of_r_),:] = 0
            else:
                print 'Full now... Add random guys'
                whole_party = set(range(self.transferred_sample.shape[0]))
                newly_add = whole_party.difference(set(choice))
                choice += list(newly_add[0: self.transferred_sample.shape[0] - len(choice)])
                break
            count += 1
        return choice

def parallel_find_p(new_simulation):
    choice = new_simulation.train(use_grad=True)
    pool = mp.Pool(5)
    print "Start pipeline"
    result = []
    for p in np.arange(0.001, 0.3, 0.01):
        result.append(pool.apply_async(run, (new_simulation, p)))
    pool.close()
    pool.join()
    # run(new_simulation, 0.01)



def run(cls_instance, i):
    cls_instance.optimizer([i])



def parallel_validate(new_simulation, i,j, dir_name):
    choice = new_simulation.train(use_grad=True)
    result = new_simulation.validate(choice, times=1)
    print result
    with open(dir_name + '/' + new_simulation.data_source + '_' +
              str(i) + '_' + str(j) + ".json", 'w') as f:
        f.write(json.dumps(result))


def simulate_pipeline(change_k=False, change_p=False, change_cand=False, change_prior=False,opt=False):
    if change_k:
        changed_paras = range(10,40,10)
        dir_name = 'change_k'
    elif change_p:
        changed_paras = [0.003, 0.05, 0.281]
        dir_name = 'change_p'
    elif change_cand:
        changed_paras = [1000,800,600,400,200]
        dir_name = 'change_cand'
    elif change_prior:
        changed_paras = [0, 0.01,0.1]
        dir_name = 'change_prior'
    else:
        dir_name = 'test'
        changed_paras = [0]

    try:
        os.listdir(dir_name)
    except OSError:
        os.mkdir(dir_name)

    dataset_num = len(changed_paras)
    iter_per_set = 5

    for i in range(0, dataset_num):
        print "Start a new set!"
        plc_num = 900
        people = 3000
        candidate = 200
        k_favor = 3
        obfuscate = 0.0
        eps = 4
        p = 0.01
        q = 1.0/((1.0-p)/(p*np.exp(eps)) + 1)
        # q=1

        d = {}

        if change_k:
            d['title'] = 'Change k'
            k_favor = changed_paras[i]
        elif change_cand:
            d['title'] = 'Change candidate num'
            candidate = changed_paras[i]
        elif change_p:
            d['title'] = 'Change p'
            p = changed_paras[i]
            q = 1.0/((1.0-p)/(p*np.exp(eps)) + 1)
        elif change_prior:
            d['title'] = 'Change p'
            obfuscate = changed_paras[i]

        d['dataset_num'] = dataset_num
        d['method_num'] = 4
        d['iter_per_set'] = iter_per_set
        d['dir_name'] = dir_name
        d['change_paras'] = changed_paras


        new_simulation = Diff_Coverage(flip_p=p, flip_q=q, candidate_num=candidate,
                                       plc_num=plc_num, people=people, k_favor=k_favor, max_iter=4000,
                                       data_src="MCS", uniform=False,freeze=False,random_start=1,
                                       obfuscate=obfuscate)

        d['data_src'] = new_simulation.data_source
        d['pic_name'] = d['data_src'] + '_' + str(eps) + '_' + d['dir_name'] + '.png'
        with open(d['dir_name'] + '/simu.json','w') as f:
            f.write(json.dumps(d))
            
        if i != 0 and change_prior == True:
            new_simulation.new_set = False
            print "Use old samples"
        else:
            new_simulation.new_set = True
            try:
                os.remove(new_simulation.sample_dir)
            except:
                print "No real sample detected!"
            try:
                os.remove(new_simulation.transfer_dir)
            except:
                print "No transferred data detected!"

        if opt:
            parallel_find_p(new_simulation)
        else:
            for j in range(0, iter_per_set):
                if j > 0:
                    new_simulation.freeze = True
                    pool = mp.Pool(5)
                    print "Start pipeline"
                    for j in range(1, iter_per_set):
                        pool.apply_async(parallel_validate, (new_simulation, i,j,dir_name))
                    pool.close()
                    pool.join()
                    break
                else:
                    choice = new_simulation.train(use_grad=True)
                    result = new_simulation.validate(choice, times=1)
                    print result
                    with open(dir_name+ '/' + new_simulation.data_source + '_' +
                              str(i) + '_' + str(j) + ".json", 'w') as f:
                        f.write(json.dumps(result))



if __name__ == '__main__':
    simulate_pipeline()
    # comb_mat = np.load('comb_mat.npy')
