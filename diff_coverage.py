from __future__ import division
from scipy.special import comb
from scipy.stats import norm
import numpy as np
import random
from matplotlib import pyplot as plt
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
import json
from read_data import *
import math

def Gaussian_Approx(X, mu, std):
    return -norm.cdf(x=(X-0.5-mu)/std) + norm.cdf(x=(X+0.5-mu)/std)

class Diff_Coverage():

    def __init__(self, uniform=True, f=0, flip_p=0.001, data_src="SG",
                 plc_num=200,people=500, candidate_num=50, k_favor=5,
                 max_iter=200, stop_iter=200, parallel=False):
        # constants in rappor
        self.f = f
        self.p = flip_p
        self.q = 1 - self.p

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

        # parallel settings
        self.parallel = parallel
        self.process_num = 1
        self.pool = mp.Pool(processes=self.process_num)

        self._is_train = False

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
        # default is a uniform prior
        self.xi = np.ones(self.plc_num) * self.k_favor / (self.plc_num + 0.0)

    def posterior(self, X, plc_id):
        N = self.candidate_num
        upper_bound = 500
        p_0 = self.p - 0.5 * self.f * (self.p - self.q)
        p_1 = self.q + 0.5 * self.f * (self.p - self.q)

        xi = self.xi[plc_id]
        sum_ = 0
        numerator = p_0 ** X * (1 - p_0) ** (N - X)
        # use Gaussian to approximate Binomial when N is large
        if N > upper_bound:
            mu = N * p_0
            std = math.sqrt(N * p_0 * (1 - p_0))
            numerator *= Gaussian_Approx(X,mu,std)
        else:
            numerator *= self.comb_mat[N, X] * (1 - xi) ** N
        for i in range(0, N + 1):
            if N > upper_bound:
                mu = N * xi
                std = math.sqrt(N * xi * (1 - xi))
                outer = Gaussian_Approx(i, mu, std)
            else:
                outer = self.comb_mat[N, i] * xi ** i * (1 - xi) ** (N - i)
            if outer == 0.0:
                continue
            inner = 0
            for m in range(max([0, X + i - N]), min([i, X]) + 1):
                # sum_ += self.comb_mat[N, i] * self.comb_mat[i, m] * p_1 ** m * (1 - p_1) ** (i - m) * \
                #     self.comb_mat[N - i, X - m] * p_0 ** (X - m) * (1 - p_0) ** (N - i - X + m) * \
                #     xi ** i * (1 - xi) ** (N - i)
                if i <= upper_bound:
                    first_part = self.comb_mat[i, m] * p_1 ** m * (1 - p_1) ** (i - m)
                else:
                    mu = i * (1 - p_1)
                    std = math.sqrt(i * p_1 * (1 - p_1))
                    first_part = Gaussian_Approx(m, mu, std)
                if N - i > upper_bound:
                    mu = (N-i) * p_0
                    std = math.sqrt((N-i) * p_0 * (1 - p_0))
                    second_part = Gaussian_Approx(X-m,mu,std)
                else:
                    second_part = self.comb_mat[N - i, X - m] * p_0 ** (X - m) * (1 - p_0) ** (N - i - X + m)
                # inner += self.comb_mat[i, m] * p_1 ** m * (1 - p_1) ** (i - m) * \
                #         self.comb_mat[N - i, X - m] * p_0 ** (X - m) * (1 - p_0) ** (N - i - X + m)
                inner += first_part * second_part
            sum_ += outer * inner
        ratio = numerator / sum_
        return ratio

    # probability distribution for artificial dataset
    def prob_dist(self):
        plcs = []
        y = np.ones(self.plc_num) * 1.0 / self.plc_num

        prob_mat = np.zeros((self.plc_num, 3))
        prob_mat[:, 0] = np.arange(0, self.plc_num)
        prob_mat[:, 1] = y
        prob_mat[:, 2] = prob_mat[:, 1].cumsum()

        for plc in range(0, self.k_favor):
            in_bit = 1
            while in_bit == 1:
                r = np.random.rand()
                plc_num = prob_mat.shape[0]
                index = -1
                for i in range(0, plc_num - 1):
                    if i == 0:
                        if prob_mat[i, 2] >= r:
                            index = prob_mat[i, 0]
                            break
                    if prob_mat[i, 2] < r and prob_mat[i + 1, 2] >= r:
                        index = prob_mat[i + 1, 0]
                        break
                if index not in plcs:
                    plcs.append(int(index))
                    in_bit = 0
                # else:
                #     print "Same found"
        return plcs

    # read real dataset from different resources
    def real_sample(self, draw=False, skew=False):
        if self.data_source == "MCS":
            rd = MCS(k_favor=self.k_favor, people=self.people)
            sample = rd.make_grid()
            np.save(self.sample_dir, sample)
            self.plc_num = rd.plc_num
            print "Total recruit: ", self.people, " from places: ", self.plc_num, sample.shape[1]
            self.update_paras()
            if not self.uniform:
                self.xi = np.sum(sample, axis=0) / (0.0 + self.people)
        elif self.data_source == "SG":
            rd = SG(k_favor=self.k_favor, people=self.people, max_id=self.plc_num)
            sample = rd.make_grid()
            np.save(self.sample_dir, sample)
            print "Total recruit: ", self.people, " from places: ", self.plc_num
            self.update_paras()
            if not self.uniform:
                self.xi = np.sum(sample, axis=0) / (0.0 + self.people)
        else:
            sample = np.zeros((self.people, self.plc_num))

            # skew: add groups of people covering the whole area
            if skew:
                partition = [0.2, 0.3, 0.5]
                self.k_favor = self.plc_num / self.candidate_num
                all_plc = list(np.random.permutation(self.plc_num))
                for i in range(0, int(self.people * partition[0])):
                    mod = i % self.candidate_num
                    sample[i,all_plc[mod * self.k_favor: (mod + 1) * self.k_favor]] = 1
                same_k_favor = random.sample(range(self.plc_num), self.k_favor * 5)
                for i in range(int(self.people * partition[0]), int(self.people * partition[1])):
                    small_partition = random.sample(same_k_favor, self.k_favor)
                    sample[i, small_partition] = 1
                for i in range(int(self.people * partition[1]), int(self.people * partition[2])):
                    index = self.prob_dist()
                    sample[i, index] = 1
            else:
                for i in range(0, self.people):
                    index = self.prob_dist()
                    sample[i, index] = np.ones(self.k_favor)
        self.update_paras()
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
    def transfer_sample(self, draw):
        sample = self.real_sample()

        for i in range(0, self.people):
            r = sample[i, :]
            r = np.array(map(lambda bit: self.flip(bit), r))[np.newaxis, :]
            sample[i, :] = r

        row_sum = np.sum(sample, axis=0)
        if draw:
            plt.plot(np.arange(0, self.plc_num), row_sum)
            plt.show()
        return sample

    # sum posterior in terms of locations
    def sum_posterior(self, sum_row):
        if self.parallel:
            results = [self.pool.apply(self.posterior, args=(int(x),))
                       for x in sum_row.tolist()]
            total_sum = np.sum(np.array(results))
        else:
            posterior_per_loc = [self.posterior(int(x), id) for id, x in enumerate(sum_row)]
            total_sum = np.sum(np.array(posterior_per_loc))
        return total_sum

    def train(self, parallel=False, use_grad=False):

        sample = self.transfer_sample(draw=False)
        self.parallel = parallel

        # first fit the linear regression coefficients
        self.posterior_regression(draw=False)

        # choice means those ids which are in the candidate set
        choice = random.sample(range(self.people), self.candidate_num)
        not_choice = list(set(range(self.people)).difference(set(choice)))

        # party means the bit arrays for those candidates
        party = sample[choice, :]
        unused = sample[not_choice, :]

        sum_row = np.sum(party, axis=0)
        total_sum = self.sum_posterior(sum_row)
        print "Start Training: Original Loss - " + str(total_sum)

        same_count = 0
        for i in range(0, self.iter):

            if use_grad:
                diff, id_of_r, id_of_r_ = self.grad_boosting(
                    sum_row, party, unused)
                r = choice[id_of_r]
                r_ = not_choice[id_of_r_]
                if diff > 0:
                    choice[id_of_r] = r_
                    not_choice[id_of_r_] = r
                    party = sample[choice, :]
                    unused = sample[not_choice, :]
                    sum_row = np.sum(party, axis=0)
                    print 'Loss reduce by: ', str(diff), 'r and r_ is: ', r, r_
                else:
                    break
            else:
                id_of_r = np.random.randint(self.candidate_num)
                r = choice[id_of_r]
                id_of_r_ = random.sample(range(len(unused)), 1)[0]
                r_ = not_choice[id_of_r_]

                sum_row_new = sum_row - party[id_of_r, :] + sample[r_, :]
                total_sum_new = self.sum_posterior(sum_row_new)

                if total_sum_new < total_sum:
                    total_sum = total_sum_new
                    choice[id_of_r] = r_
                    not_choice[id_of_r_] = r
                    party = sample[choice, :]
                    unused = sample[not_choice, :]

                    sum_row = np.sum(party, axis=0)
                    same_count = 0
                else:
                    same_count += 1

                print str(total_sum), same_count

                if same_count > 1 and use_grad:
                    break
                elif same_count > self.stop_iter:
                    break
        total_sum = self.sum_posterior(sum_row)
        print 'Finish:', total_sum
        np.save(self.transfer_dir, sample)
        self._is_train = True
        return choice

    def validate(self, choice, times=10):
        if not self._is_train:
            exit(1)
        fail = 0
        abnormal = 0
        true_sample = np.load(self.sample_dir)
        fake_sample = np.load(self.transfer_dir)

        result = []
        true_opt_party = true_sample[choice, :]
        fake_opt_party = fake_sample[choice, :]
        sum_row = np.sum(fake_opt_party, axis=0)
        total_sum_opt = self.sum_posterior(sum_row)
        coverage_opt = np.sum(np.sum(true_opt_party, axis=0) > 0)
        result.append((coverage_opt, total_sum_opt))
        for i in range(times):
            rand_choice = random.sample(range(self.people), self.candidate_num)
            true_rand_party = true_sample[rand_choice, :]
            fake_rand_party = fake_sample[rand_choice, :]

            # print out the posterior
            sum_row = np.sum(fake_rand_party, axis=0)
            total_sum_rand = self.sum_posterior(sum_row)
            coverage_rand = np.sum(np.sum(true_rand_party, axis=0) > 0)
            result.append((coverage_rand, total_sum_rand))

            # print "posterior probability: ", total_sum_rand
            # print "random select: ", coverage_rand

            # print "posterior probability: ", total_sum_opt
            # print "true select: ", coverage_opt
            # print "*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*"

            if coverage_rand > coverage_opt:
                fail += 1
                if total_sum_rand > total_sum_opt:
                    abnormal += 1

        print "abnormal rate: " + str(abnormal) + '/' + str(times)
        print "fail rate: " + str(fail) + '/' + str(times)
        # first element is opt, others are rand
        return result

    # plot random select results to find relationship between loss function and target
    def random_select(self, random_choice=50, file_name='p_0.01'):
        true_sample = np.load(self.sample_dir)
        fake_sample = np.load(self.transfer_dir)
        rand_prob_ = np.zeros(random_choice)
        rand_region_ = np.zeros(random_choice)

        for i in range(0, random_choice):
            self.iter = np.random.randint(1000, 1500)
            self.stop_iter = np.random.randint(50, 100)

            rand_choice = random.sample(range(self.people), self.candidate_num)
            # choice = self.train()
            true_rand_party = true_sample[rand_choice, :]
            fake_rand_party = fake_sample[rand_choice, :]

            # print out the posterior
            sum_row = np.sum(fake_rand_party, axis=0)
            total_sum_rand = self.sum_posterior(sum_row)
            rand_prob_[i] = total_sum_rand
            coverage_rand = np.sum(np.sum(true_rand_party, axis=0) > 0)
            rand_region_[i] = coverage_rand

            if i % 10 == 0:
                print i

        # plt.scatter(real_prob_, real_region,c='r')
        plt.scatter(rand_prob_, rand_region_, c='b')
        plt.xlabel("Sum of posterior over location")
        plt.ylabel("Coverage")
        title = "transfer_prob_=" + str(self.p) + ", candidate_num=" + str(
            self.candidate_num) + ", k_favor=" + str(self.k_favor)
        plt.title(title)
        plt.savefig('img/' + file_name + '.png')
        print self.p, self.k_favor, self.candidate_num

    def posterior_regression(self, draw=False):
        # now coefficients should be a matrix, each place has its own slope and intercept

        # only take several points to fit the linear regression,
        # avoiding high cost of iteration

        # x_max = self.candidate_num + 1
        x_max = 10
        for plc_id in range(self.plc_num):
            X_range = np.arange(0, x_max)
            ans = np.zeros(X_range.shape)
            stop_count = self.candidate_num + 1
            for X in X_range:
                ans[X] = np.log(self.posterior(X=X,plc_id=plc_id))
                if ans[X] == -np.inf or ans[X] == np.nan:
                    ans[X] = 0
                    stop_count = X
                    break
            X_range = X_range[0:stop_count, np.newaxis]
            ans = ans[0:stop_count, np.newaxis]
            if(draw):
                plt.scatter(X_range, ans)
                plt.show()
            lr = LinearRegression()
            lr.fit(X_range, ans)
            if self.uniform:
                # one fit is enough
                self.a = np.ones(self.a.shape) * lr.coef_
                self.b = np.ones(self.b.shape) * lr.intercept_
                break
            else:
                self.a[plc_id] = lr.coef_
                self.b[plc_id] = lr.intercept_
            # print "Regression Score: ", lr.score(X_range, ans)
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


# tool for examining simulation results
def stat_compute():
    dataset_num = 20
    iter_per_set = 20
    real = 'real_result/'
    simu = 'simulation/'
    dir_ = simu
    true_means = np.zeros((dataset_num,))
    true_stds = np.zeros((dataset_num,))
    rand_means = np.zeros((dataset_num,))
    rand_stds = np.zeros((dataset_num,))
    file_list = ['p_0.1/','p_0.01/','p_0.001/']
    whole_mean = np.zeros((len(file_list),2))
    whole_std = np.zeros((len(file_list),2))
    for count, fn in enumerate(file_list):
        whole_true = []
        whole_rand = []
        for i in range(0, dataset_num):
            true = []
            rand = []
            for j in range(0, iter_per_set):
                with open(dir_ + fn + str(i) + '_' + str(j) + ".json",'r') as f:
                    results = json.loads(f.read())
                    true.append(results[0][0])
                    whole_true.append((results[0]))
                    rand += [i[0] for i in results[1:]]
                    whole_rand += [i[0] for i in results[1:]]
            true_means[i] = np.mean(np.array(true))
            true_stds[i] = np.std(np.array(true))
            rand_means[i] = np.mean(np.array(rand))
            rand_stds[i] = np.std(np.array(rand))

        whole_mean[count, 0] = np.mean(np.array(whole_true))
        whole_std[count, 0] = np.std(np.array(whole_true))
        whole_mean[count, 1] = np.mean(np.array(whole_rand))
        whole_std[count, 1] = np.std(np.array(whole_rand))

    ind = np.arange(len(file_list))
    width = 0.3
    p1 = plt.bar(ind - width / 2.0, whole_mean[:,0], width=width, color='c', yerr=whole_std[:,0])
    p2 = plt.bar(ind + width / 2.0, whole_mean[:,1], width=width, color='r', yerr=whole_std[:,1])
    marker = [i.rstrip('/') for i in file_list]
    plt.xticks(ind, marker)
    plt.legend((p1[0], p2[0]),("Greedy (With Gradient Boosting)", "Random"))
    plt.ylabel("Num of Covered Areas")
    plt.show()


def simulate_pipeline(real=True):
    dataset_num = 7
    k_favor_list = range(5,40,5)
    iter_per_set = 10
    for i in range(0, dataset_num):
        k_favor = k_favor_list[i]
        new_simulation = Diff_Coverage(flip_p=0.1, candidate_num=300,
                                       plc_num=2000, people=1000, k_favor=k_favor, max_iter=400)
        # new_simulation.real_sample(draw=True, skew=True, real=False)
        # new_simulation.transfer_sample(draw=False)
        # new_simulation.posterior_regression(draw=True)

        for j in range(0, iter_per_set):
            choice = new_simulation.train(use_grad=True)
            result = new_simulation.validate(choice, times=2)
            # print result
            with open(str(i) + '_' + str(j) + ".json", 'w') as f:
                f.write(json.dumps(result, indent=4))



if __name__ == '__main__':
    simulate_pipeline()