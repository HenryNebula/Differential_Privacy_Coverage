from __future__ import division
from scipy.special import comb
from scipy.stats import norm
import os
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression
from read_data import *
import math
from numpy.linalg import lstsq
import functools
# from util import *

def unwrap_self_find_coeffs(arg, **kwarg):
    return Diff_Coverage.find_coeffs(*arg, **kwarg)

def Gaussian_Approx(X, mu, std):
    if std == 0 and X == 0:
        return 1
    elif std == 0:
        return 0
    else:
        return -norm.cdf(x=X-0.5, loc=mu, scale=std) + norm.cdf(x=X+0.5, loc=mu, scale=std)

class Diff_Coverage():

    def __init__(self, uniform=True, f=0, flip_p=0.001, flip_q = 0.1, data_src="SG",
                 plc_num=200,people=500, candidate_num=50, k_favor=5,
                 max_iter=200,random_start=5,obfuscate=0, pool_num=2):
        # constants in rappor
        self.f = f
        self.p = flip_p
        self.q = flip_q
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

        # paras for regression: a for slope, b for intersect
        # vector, different
        self.a = None
        self.b = None
        self.prior_dict = {}

        self.prob_mat = None
        self.random_start = random_start
        self.obfuscate = obfuscate

    # make table for combination numbers
    def make_table(self):
        comb_mat = np.zeros((self.candidate_num + 1, self.candidate_num + 1))
        for i in range(0, self.candidate_num + 1):
            for j in range(0, i + 1):
                comb_mat[i][j] = comb(i, j)
        np.save('comb_mat', comb_mat)

    def posterior_core(self, X, xi):
        p_0 = self.p - 0.5 * self.f * (self.p - self.q)
        p_1 = self.q + 0.5 * self.f * (self.p - self.q)
        N = self.candidate_num
        comb_mat = self.comb_mat
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

    def update_prior(self, sample):
        if self.uniform:
            self.xi = np.ones(self.plc_num) * self.k_favor / (self.plc_num + 0.0)
            self.prior_dict[self.xi[0]] = 0
        else:
            self.xi = np.sum(sample, axis=0) / self.people

        self.xi = self.xi + np.random.rand(self.xi.shape[0]) * self.obfuscate
        self.xi[self.xi < 0] = 0.0
        self.xi[self.xi > 1] = 1

    def real_sample(self, draw=False):
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

        print "Total recruit: ", self.people, " from places: ", self.plc_num
        self.update_prior(sample)
        if draw:
            row_sum = np.sum(sample, axis=0)
            plt.plot(np.arange(0, self.plc_num), row_sum)
            plt.show()
        return sample

    # create "fake" dataset after random response
    def transfer_sample(self, sample, draw=False):
        trans_sample = np.copy(sample)
        def flip(bit):
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

        for i in range(0, self.people):
            r = trans_sample[i, :]
            r = np.array(map(lambda bit: flip(bit), r))[np.newaxis, :]
            trans_sample[i, :] = r

        if draw:
            row_sum = np.sum(trans_sample, axis=0)
            plt.plot(np.arange(0, self.plc_num), row_sum)
            plt.show()

        return trans_sample

    # sum posterior in terms of locations
    def sum_posterior(self, sum_row):
        if self.uniform and self.prob_mat is not None:
            posterior_per_loc = [self.prob_mat[int(x)] for x in sum_row]
        else:
            posterior_per_loc = [self.posterior_core(int(x), self.xi[id]) for id, x in enumerate(sum_row)]
        total_sum = np.sum(np.array(posterior_per_loc))
        return total_sum

    def find_coeffs(self,xi):
            # print xi
            x_max = 10
            X_range = np.arange(0, x_max)
            stop_count = self.candidate_num + 1
            ans = np.zeros(X_range.shape)
            for X in X_range:
                ans[X] = np.log(self.posterior_core(X, xi))
                if np.isinf(ans[X]) or np.isnan(ans[X]):
                    ans[X] = 0
                    stop_count = X
                    break
            X_range = X_range[0:stop_count, np.newaxis]
            ans = ans[0:stop_count, np.newaxis]
            if ans.shape[0] == 1:
                coef = 0.0, 0.0
            else:
                X_range = np.hstack([X_range, np.ones((len(X_range),1))])
                # lr = LinearRegression()
                # lr.fit(X_range, ans)
                # coef = lr.coef_, lr.intercept_
                coef = lstsq(X_range, ans)[0]
            return coef

    def posterior_regression(self, draw=False):
        # now coefficients should be a matrix, each place has its own slope and intercept
        x_max = 10
        self.a = np.zeros(self.plc_num)
        self.b = np.zeros(self.plc_num)
        if self.uniform:
            # print "Make posterior tables!"
            self.prob_mat = np.array([self.posterior_core(x, self.xi[1]) for x in range(self.candidate_num + 1)])
            self.prob_mat[np.isnan(self.prob_mat)] = 0
            coeffs = self.find_coeffs(self.xi[1])
            self.a = self.a + coeffs[0]
            self.b = self.b * coeffs[1]
        else:
            prior_dict = {}
            plc_index = np.zeros(self.plc_num)
            prior_list = []
            count = 0
            for plc_id in range(self.plc_num):
                xi = self.xi[plc_id]
                if not prior_dict.has_key(xi):
                    prior_dict[xi] = count
                    prior_list.append(xi)
                    count += 1
                plc_index[plc_id] = prior_dict[xi]
            pool = Pool(processes=5)
            coef_list = pool.map(unwrap_self_find_coeffs, zip([self]*len(prior_list), prior_list))
            for plc_id in range(self.plc_num):
                coef = coef_list[int(plc_index[plc_id])]
                self.a[plc_id] = coef[0]
                self.b[plc_id] = coef[1]
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

    def intelli_grad(self, trans_sample):
        sum_row = 0
        count = 0
        choice = []
        while count <= self.candidate_num:
            # grad_ = abs(self.a * np.exp(self.b) * np.exp(self.a * sum_row)).T
            grad_ = abs((np.exp(self.a) - 1) * np.exp(self.a * sum_row + self.b)).T
            candidate_add = np.dot(trans_sample, grad_)
            id_of_r_ = np.argmax(candidate_add)
            max_val = np.max(candidate_add)
            if max_val != 0:
                choice.append(id_of_r_)
                sum_row += trans_sample[int(id_of_r_), :]
                trans_sample[int(id_of_r_),:] = 0
            else:
                print 'Full now... Add random guys'
                whole_party = set(range(trans_sample.shape[0]))
                newly_add = whole_party.difference(set(choice))
                choice += list(newly_add[0: trans_sample.shape[0] - len(choice)])
                break
            count += 1
        return choice

    def train(self, trans_sample, use_grad=True,):
        # first fit the linear regression coefficients
        self.posterior_regression(draw=False)

        min_loss = 1e10
        final_choice = []
        for times in range(0, self.random_start):
            # choice means those ids which are in the candidate set
            choice = random.sample(range(self.people), self.candidate_num)
            not_choice = list(set(range(self.people)).difference(set(choice)))

            # party means the bit arrays for those candidates
            party = trans_sample[choice, :]
            unused = trans_sample[not_choice, :]

            sum_row = np.sum(party, axis=0)
            # total_sum = self.sum_posterior(sum_row)
            # print "Start Training: Original Loss - " + str(total_sum)

            same_count = 0
            past_set = set()
            for i in range(0, self.iter):
                if use_grad:
                    diff, id_of_r, id_of_r_ = self.grad_boosting(sum_row, party, unused)
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
                        party = trans_sample[choice, :]
                        unused = trans_sample[not_choice, :]
                        sum_row = np.sum(party, axis=0)
                        # print 'Loss reduce by: ', str(diff), 'r and r_ is: ', r, r_
                    else:
                        break
                else:
                    choice = self.intelli_grad(trans_sample)
            # total_sum = self.sum_posterior(sum_row)
            total_sum = 0
            print 'Finish:', total_sum, same_count, "Random start: ", times
            if total_sum < min_loss:
                min_loss = total_sum
                final_choice = choice
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
    def validate(self,true_sample, trans_sample, choice, times=1):
        result = []
        print 'Perfect Baseline'
        max_sum, final_choice = self.perfect_baseline(true_sample)
        result.append((max_sum, len(np.where(self.xi > 0)[0])))

        true_opt_party = true_sample[choice, :]
        fake_opt_party = trans_sample[choice, :]
        sum_row = np.sum(fake_opt_party, axis=0)
        # total_sum_opt = self.sum_posterior(sum_row)
        total_sum_opt = 0
        coverage_opt = np.sum(np.sum(true_opt_party, axis=0) > 0)
        result.append((coverage_opt, total_sum_opt))
        for i in range(times):
            max_sum, final_choice = self.perfect_baseline(trans_sample)
            final_party = true_sample[final_choice, :]
            real_sum = np.sum(np.sum(final_party, axis=0) > 0)
            result.append((real_sum, max_sum))

            rand_choice = random.sample(range(self.people), self.candidate_num)
            true_rand_party = true_sample[rand_choice, :]
            fake_rand_party = trans_sample[rand_choice, :]

            # print out the posterior
            sum_row = np.sum(fake_rand_party, axis=0)
            # total_sum_rand = self.sum_posterior(sum_row)
            total_sum_rand = 0
            coverage_rand = np.sum(np.sum(true_rand_party, axis=0) > 0)
            result.append((coverage_rand, total_sum_rand))

        # first element is opt, second is random-greedy, third is fake-data
        return result


def simulate_pipeline(change_k=False, change_p=False, change_cand=False, change_prior=False, opt=False):
    if change_k:
        changed_paras = range(10, 40, 10)
        dir_name = 'change_k'
    elif change_p:
        changed_paras = [0.003, 0.05, 0.281]
        dir_name = 'change_p'
    elif change_cand:
        changed_paras = [1000, 800, 600, 400, 200]
        dir_name = 'change_cand'
    elif change_prior:
        changed_paras = [0, 0.01, 0.1]
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

    plc_num = 900
    people = 5000
    candidate = 2000
    k_favor = 3
    obfuscate = 0.0
    eps = 4
    p = 0.01
    q = 1.0 / ((1.0 - p) / (p * np.exp(eps)) + 1)
    data_src = 'MCS'

    d = {'title': dir_name, 'dataset_num': dataset_num, 'method_num': 4, 'iter_per_set': iter_per_set,
         'dir_name': dir_name, 'change_paras': changed_paras, 'data_src': data_src}
    d['pic_name'] = d['data_src'] + '_' + str(eps) + '_' + d['dir_name'] + '.png'
    with open(d['dir_name'] + '/simu.json', 'w') as f:
        f.write(json.dumps(d))

    for i in range(0, dataset_num):
        print "Start a new set!"
        if change_k:
            k_favor = changed_paras[i]
        elif change_cand:
            candidate = changed_paras[i]
        elif change_p:
            p = changed_paras[i]
            q = 1.0 / ((1.0 - p) / (p * np.exp(eps)) + 1)
        elif change_prior:
            obfuscate = changed_paras[i]

        new_dataset = (i == 0) or ((i > 0) and (change_k or change_cand))
        old_set_new_trans = (i > 0) and change_p
        new_simulation = Diff_Coverage(flip_p=p, flip_q=q, candidate_num=candidate,
                                       plc_num=plc_num, people=people, k_favor=k_favor, max_iter=4000,
                                       data_src=data_src, uniform=False, random_start=1, obfuscate=obfuscate)
        if new_dataset:
            true_sample = new_simulation.real_sample()
            trans_sample = new_simulation.transfer_sample(true_sample)
        else:
            if old_set_new_trans:
                trans_sample = new_simulation.transfer_sample(true_sample)
        if obfuscate:
            new_simulation.update_prior(true_sample)

        for j in range(iter_per_set):
            choice = new_simulation.train(trans_sample=trans_sample)
            result = new_simulation.validate(true_sample, trans_sample, choice, times=1)
            print result
            with open(dir_name + '/' + new_simulation.data_source + '_' +
                      str(i) + '_' + str(j) + ".json", 'w') as f:
                f.write(json.dumps(result))


if __name__ == '__main__':
    simulate_pipeline()

