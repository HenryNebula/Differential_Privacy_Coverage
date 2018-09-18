from __future__ import division
import matplotlib
matplotlib.use("Agg")
from scipy.special import comb
from scipy.stats import norm, entropy
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import Lasso
from sklearn.feature_selection import f_regression
from read_data import *
import math
from numpy.linalg import lstsq
from datetime import datetime
import scipy.sparse as sparse
import functools

def unwrap_self_find_coeffs(arg, **kwarg):
    return Diff_Coverage.find_coeffs(*arg, **kwarg)


def unwrap_self_posterior_core(arg, **kwarg):
    return Diff_Coverage.posterior_core(*arg, **kwarg)

def Gaussian_Approx(X, mu, std):
    if std == 0 and X == 0:
        return 1
    elif std == 0:
        return 0
    else:
        return -norm.cdf(x=X-0.5, loc=mu, scale=std) + norm.cdf(x=X+0.5, loc=mu, scale=std)

class Diff_Coverage():

    def __init__(self, flip_p, flip_q, data_src,
                 target_constraint, candidate_num, k_favor, hist=False,
                 max_iter=4000, random_start=5, obfuscate=0,
                 uniform=False, f=0):
        # constants in rappor
        self.f = f
        self.p = flip_p
        self.q = flip_q
        
        # parameters in crowdsourcing
        self.target_constraint = target_constraint
        self.k_favor = k_favor
        self.candidate_num = candidate_num
        self.enable_hist = hist

        # optimization options
        self.iter = max_iter
        self.random_start = random_start
        self.obfuscate = obfuscate

        # useless parameters
        self.uniform = uniform
        self.xi = None
        self.data_source = data_src

        # pre-computes and placeholders
        self.comb_mat = np.load(pre_compute_dir + 'comb_mat.npy')
        self.a = None # paras for regression: a for slope,
        self.b = None # b for intersect

        self.people = 0
        self.target_num = 0
        self.nnz_target_num = 0
        self.compressed_sample = None
        self.raw_sample = None
        self.perturbed_sample = None
        self.prior_dict = {}
        self.histeq_dict = {} # remap_id: ([(contains original locs)], sum_of_prior)


        self.nnz_array = None
        self.overlap_array = None
        self.prob_mat = None

    # make table for combination numbers
    def make_table(self):
        comb_mat = np.zeros((1000 + 1, 1000 + 1))
        for i in range(0, self.candidate_num + 1):
            for j in range(0, i + 1):
                comb_mat[i][j] = comb(i, j)
        np.save(pre_compute_dir + 'comb_mat', comb_mat)

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
        def histeq():
            L = self.target_num
            cdf = np.cumsum(self.xi) / np.sum(self.xi)
            level = np.ceil((L-1)*cdf + 0.5) - 1

            new_pdf = np.zeros((L,1))

            for i, l in enumerate(level):
                l = int(l)
                # TODO: First map(histeq) and Second Map(delete zeros)
                new_pdf[l] += self.xi[i]
                if l not in self.histeq_dict:
                    self.histeq_dict[l] = [[],0]
                self.histeq_dict[l][1] = new_pdf[l]
                self.histeq_dict[l][0].append(i)

            self.nnz_array = np.where(new_pdf!=0)[0]
            self.overlap_array = np.array([min(len(self.histeq_dict[l][0]), 100) for l in self.nnz_array])
            self.xi = new_pdf[new_pdf!=0]
            self.nnz_target_num = len(self.xi)

        if self.uniform:
            self.xi = np.ones(self.target_num) * self.k_favor / (self.target_num + 0.0)
            self.prior_dict[self.xi[0]] = 0
        else:
            self.xi = np.sum(sample, axis=0) / self.people
        print "mean of prior:{0:3.6f}, sum:{1:2.2f}, std:{2:3.5f}".format(np.mean(self.xi),
                                                                          np.sum(self.xi),
                                                                          np.std(self.xi))
        self.xi = self.xi + np.random.rand(self.xi.shape[0]) * self.obfuscate * np.mean(self.xi)
        self.xi[self.xi < 0] = 0.0
        self.xi[self.xi > 1] = 1
        if self.enable_hist:
            histeq()
        else:
            self.overlap_array = np.ones(self.xi.shape)

    def real_sample(self, draw=False):
        if self.data_source == "MCS":
            rd = MCS(k_favor=self.k_favor, x_granu=self.target_constraint, y_granu=self.target_constraint)
            sample = rd.read_scratch(draw=draw)
            self.people, self.target_num = sample.shape

        elif self.data_source == "SG":
            rd = SG(k_favor=self.k_favor, max_id=self.target_constraint)
            sample = rd.make_grid()
            self.people, self.target_num = sample.shape
        else:
            rd = CG(k_favor=self.k_favor, people=self.people, max_id=self.target_constraint)
            sample = rd.make_grid()

        print "Data source:{0}, p:{1}, people:{2}, candidate:{3}, target: {4}, k_favor:{5}".format(
            self.data_source, self.p, self.people, self.candidate_num, self.target_num, self.k_favor)

        self.update_prior(sample)
        self.raw_sample = sample

        if self.enable_hist:
            self.compressed_sample = np.zeros((sample.shape[0], len(self.nnz_array)))
            #TODO: change sample columns
            for i, col in enumerate(self.nnz_array):
                aggregate_cols = self.histeq_dict[col][0]
                self.compressed_sample[:, i] = np.sum(sample[:, aggregate_cols], axis=1)
            self.compressed_sample = np.array(self.compressed_sample, dtype=bool)
        else:
            self.compressed_sample = self.raw_sample
        if draw:
            row_sum = np.sum(sample, axis=0)
            idx = range(np.max(row_sum)+1)
            count = [0] * len(idx)
            for s in row_sum:
                count[s] += 1
            plt.loglog(idx, count)
            print 'plot it'
            plt.savefig('self-xi.png')

    # create "fake" dataset after random response
    def transfer_sample(self, draw=False):
        self.perturbed_sample = np.copy(self.compressed_sample)
        shape = self.perturbed_sample.shape
        print(shape)
        if self.f != 0:
            rand_mtx = np.random.rand(shape[0], shape[1])
            self.perturbed_sample[rand_mtx < 0.5 * self.f] = True
            self.perturbed_sample[(rand_mtx < self.f) & (rand_mtx >= 0.5 * self.f)] = False

        rand_mtx = np.random.rand(shape[0], shape[1])
        self.perturbed_sample[(rand_mtx >= self.q) & self.perturbed_sample] = False
        self.perturbed_sample[(rand_mtx < self.p) & (~self.perturbed_sample)] = True

        if draw:
            row_sum = np.sum(self.perturbed_sample, axis=0)
            plt.plot(np.arange(0, self.target_num), row_sum)
            plt.show()

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
                coef = lstsq(X_range, ans, rcond=None)[0]
            return coef

    def posterior_regression(self, draw=False):
        target_num = self.compressed_sample.shape[1]
        # now coefficients should be a matrix, each place has its own slope and intercept
        self.a = np.zeros(target_num)
        self.b = np.zeros(target_num)
        # if self.uniform:
        #     # print "Make posterior tables!"
        #     self.prob_mat = np.array([self.posterior_core(x, self.xi[1]) for x in range(self.candidate_num + 1)])
        #     self.prob_mat[np.isnan(self.prob_mat)] = 0
        #     coeffs = self.find_coeffs(self.xi[1])
        #     self.a = self.a + coeffs[0]
        #     self.b = self.b * coeffs[1]
        # else:
        prior_dict = {}
        plc_index = np.zeros(target_num)
        prior_list = []
        count = 0
        for plc_id in range(target_num):
            xi = self.xi[plc_id]
            if not prior_dict.has_key(xi):
                prior_dict[xi] = count
                prior_list.append(xi)
                count += 1
            plc_index[plc_id] = prior_dict[xi]
        # print "Total different prior: ", len(prior_dict)
        pool = Pool(processes=cpu_count() * 2)
        coef_list = pool.map(unwrap_self_find_coeffs, zip([self]*len(prior_list), prior_list))
        for plc_id in range(target_num):
            coef = coef_list[int(plc_index[plc_id])]
            self.a[plc_id] = coef[0]
            self.b[plc_id] = coef[1]
        print "Linear Regression over"

    def grad_boosting(self, sum_row, party, unused, optimizer, top_k=1):

        def coverage_gain(visit_mtx, sum_row, add):
            Ind = []
            for i in range(visit_mtx.shape[0]):
                Ind.append(np.where(visit_mtx[i,:]))

            part_a = np.array([self.a[i] for i in Ind])
            part_b = np.array([self.b[i] for i in Ind])
            part_x = np.array([sum_row[i] for i in Ind])
            term = -1 if not add else 1
            coeff = term * part_a
            exp_part = part_a * part_x + part_b
            grad_ = []
            for i, term in enumerate(coeff):
                grad_.append(np.sum((np.exp(coeff[i]) - 1) * np.exp(exp_part[i])))
            grad_ = np.array([grad_])
            if top_k == 1:
                id_of_r = [np.argmin(grad_)]
            else:
                id_of_r = np.argpartition(grad_.squeeze(), top_k)[:top_k]
            return id_of_r

        def gradient(visit_mtx, sum_row, add):
            sparse_party = sparse.csr_matrix(visit_mtx)
            grad_ = np.array(self.overlap_array.squeeze(), dtype=np.float64)
            grad_ *= abs(self.a * np.exp(self.b) * np.exp(self.a * sum_row))
            grad_ = grad_.T
            grad_ = sparse.csr_matrix(grad_)
            term = 1 if not add else -1
            candidate_sub = term * sparse_party * grad_.T
            if top_k == 1:
                id_of_r = [np.argmin(candidate_sub)]
            else:
                id_of_r = np.argpartition(candidate_sub.toarray().squeeze(), top_k)[:top_k]
            return id_of_r

        def handler(optimizer, sum_row, party):
            if optimizer not in ['gradient', 'coverage_gain']:
                print('Wrong Optimizer {}'.format(optimizer))
                exit(1)
            if optimizer == 'gradient':
                optimizer = gradient
            else:
                optimizer = coverage_gain
            party = party.copy()
            id_of_r = optimizer(party, sum_row, False)
            party[id_of_r, :] = 0
            sum_row = np.sum(party, axis=0)
            id_of_r_ = optimizer(unused, sum_row, True)
            return np.array(id_of_r, dtype=np.int32), np.array(id_of_r_, dtype=np.int32)

        return handler(optimizer, sum_row, party)

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
                newly_add = list(whole_party.difference(set(choice)))
                choice += newly_add[0: trans_sample.shape[0] - len(choice)]
                break
            count += 1
        return choice

    def fast_post_sum(self, sum_, fast_mode=True):
        if fast_mode:
            return np.sum(np.exp(self.a * sum_ + self.b))
        else:
            return self.sum_posterior(sum_)

    def train(self, use_grad=True, optimizer='gradient', freeze=False):
        if not freeze:
            self.real_sample()
            self.transfer_sample()
            # first fit the linear regression coefficients
            self.posterior_regression(draw=False)
        trans_sample = self.perturbed_sample

        min_loss = 1e10
        final_choice = []
        for times in range(0, self.random_start):
            # choice means those ids which are in the candidate set
            choice = random.sample(range(self.people), self.candidate_num)
            not_choice = list(set(range(self.people)).difference(set(choice)))

            choice, not_choice = np.array(choice), np.array(not_choice)

            # party means the bit arrays for those candidates
            party = trans_sample[choice, :]
            unused = trans_sample[not_choice, :]

            sum_row = np.sum(party, axis=0)
            # total_sum = self.sum_posterior(sum_row)
            # print "Start Training: Original Loss - " + str(total_sum)

            same_count, increase_count = 0, 0
            past_tup = ()
            if use_grad:
                for i in range(0, self.iter):
                    id_of_r, id_of_r_ = self.grad_boosting(sum_row, party, unused, optimizer)
                    r = choice[id_of_r]
                    r_ = not_choice[id_of_r_]

                    new_sum_row = sum_row - trans_sample[r, :] + trans_sample[r_, :]
                    diff = self.fast_post_sum(sum_row) - self.fast_post_sum(new_sum_row)
                    if diff > 0:
                        if (set(r), set(r_)) == past_tup or (set(r_), set(r)) == past_tup:
                            same_count += 1
                        else:
                            past_tup = (set(r), set(r_))
                            same_count = 0
                        if same_count >= 3:
                            # print "Stop iteration at: ", i
                            break
                        increase_count = 0
                    else:
                        increase_count += 1
                        if increase_count > 5:
                            print('Can not decrease Expected Loss any more after iteration {}'.format(i))
                            break
                    choice[id_of_r] = r_
                    not_choice[id_of_r_] = r
                    party = trans_sample[choice, :]
                    unused = trans_sample[not_choice, :]
                    sum_row = np.sum(party, axis=0)

            else:
                # (x, prior): posterior
                tup_dict = {}
                def parallel_descent(sum_row):
                    keys = zip(sum_row, self.xi)
                    values = np.array([tup_dict[k] if tup_dict.has_key(k) else -1 for k in keys ])
                    null_pos = np.where(values == -1)[0]

                    pool = Pool(cpu_count() * 2)
                    new_posterior = pool.map(unwrap_self_posterior_core, zip([self]*len(null_pos),
                                                                sum_row[null_pos] ,self.xi[null_pos]))

                    values[null_pos] = np.array(new_posterior)
                    for i, pos in enumerate(null_pos):
                        tup_dict[keys[pos]] = new_posterior[i]

                    return values

                while 1:
                    sum_row -= 1
                    values = parallel_descent(sum_row)
                    min_choice = np.argmin(values)

                    sum_row += 1

                    values = parallel_descent(sum_row)

            # total_sum = self.sum_posterior(sum_row)
            total_sum = 0
            print 'Finish Iter_{}'.format(times)
            if total_sum < min_loss:
                min_loss = total_sum
                final_choice = choice
        return final_choice

    def perfect_baseline(self, ts, baseline='perfect'):
        plcs = set()
        final_choice = []
        count = 0
        old_sum = 0
        true_sample = sparse.lil_matrix(ts.copy())
        while count <= self.candidate_num:
            # if count % 100 == 0:
                # print '[{0}] Iteration:{1}'.format(datetime.now(), count)
            person_contrib = true_sample.sum(axis=1)
            id_ = np.argmax(person_contrib)
            new_plcs = true_sample[int(id_), :].tocsr()
            new_plcs = new_plcs.indices
            [plcs.add(p) for p in new_plcs]
            true_sample[:, new_plcs] = 0
            count = count + 1
            if old_sum == len(plcs):
                print '{} only needs: {}'.format(baseline, count)
                whole_group = set(range(true_sample.shape[0]))
                newly_add = whole_group.difference(set(final_choice))
                final_choice += list(newly_add)[0:self.candidate_num - len(final_choice)]
                break
            else:
                final_choice.append(id_)
                old_sum = len(plcs)
        max_sum = len(plcs)
        return max_sum, final_choice

    def rappor_baseline(self, alpha=0.1):

        def lasso_sig(trans_sample, lasso=True):
            N,M = trans_sample.shape
            sum_row = np.sum(trans_sample, axis=0)
            y = sum_row - (self.p + 0.5 * self.f * (self.q - self.p)) * N
            y /= (1 - self.f) * (self.q - self.p)
            X = np.eye(M)
            
            if lasso:
                lasso = Lasso(alpha=alpha, )
                lasso.fit(X, y)
                coef = lasso.sparse_coef_
                if len(coef.indices):
                    X = X[coef.indices, :]
                    y = y[coef.indices]
                    _, pval = f_regression(X, y)
                    significants = len(np.where(pval>0.05/M)[0])
                else:
                    significants = 0
            else:
                _, pval = f_regression(X, y)
                significants = len(np.where(pval > 0.05 / M)[0])
            return significants

        max_sig = -1
        choice = {np.random.randint(0, self.perturbed_sample.shape[0] - 1)}
        unused = set(range(self.perturbed_sample.shape[0])).difference(choice)
        for i in range(self.candidate_num-1):
            print "LASSO {}".format(i)
            newly_add = -1
            for user in unused:
                temp_choice = list(choice)
                temp_choice.append(user)
                trans_ = self.perturbed_sample[temp_choice, :]
                significants = lasso_sig(trans_)
                if significants >= max_sig:
                    max_sig = significants
                    newly_add = user
            if newly_add != -1:
                choice.add(newly_add)
                unused.difference({newly_add})
            else:
                break
        if len(choice) < self.candidate_num:
            unused = list(unused)
            choice = choice.union(set(unused[:self.candidate_num - len(choice)]))

        return 0,list(choice)


    # compare with two baseline: use "fake data" directly and greedy random search
    def validate(self, choice, times=1, rappor=False):
        true_sample = self.raw_sample
        trans_sample = self.perturbed_sample

        def plot_by_loc(choice_list):
            def plot_core(ax, sum_, legend, max_=50):
                # sum_ = sorted(sum_, reverse=True)
                ax.plot(range(self.target_num), sum_,
                                 label=legend + '_{0:.3f}'.format(entropy(sum_, uniform)), )
                ax.set_ylim((0, max_))
                ax.legend()

            legends = ['perfect','optim', 'noisy','random']
            uniform = np.array([1] * self.target_num)
            fig, (real_axs, post_axs, trans_axs) = plt.subplots(nrows=3, ncols=4)

            for i, choice in enumerate(choice_list):
                real_sum = np.sum(true_sample[choice,:], axis=0)
                trans_sum = np.sum(trans_sample[choice,:], axis=0)
                post = np.exp(self.a * trans_sum + self.b)

                plot_core(real_axs[i], real_sum, legends[i])
                plot_core(post_axs[i], post, legends[i],max_=1)
                plot_core(trans_axs[i], trans_sum, legends[i])


            # plt.tight_layout()
            plt.show()

        result = []
        choice_list = []

        max_sum, final_choice = self.perfect_baseline(true_sample)
        choice_list.append(final_choice)
        result.append((max_sum, self.fast_post_sum(np.sum(trans_sample[final_choice, :], 0), fast_mode=True)))

        true_opt_party = true_sample[choice, :]
        fake_opt_party = trans_sample[choice, :]

        choice_list.append(choice)

        sum_row = np.sum(fake_opt_party, axis=0)
        # total_sum_opt = self.sum_posterior(sum_row)
        total_sum_opt = self.fast_post_sum(sum_row, fast_mode=True)
        coverage_opt = np.sum(np.sum(true_opt_party, axis=0) > 0)
        result.append((coverage_opt, total_sum_opt))

        noisy_result, noisy_post = [], []
        random_result, random_post = [], []
        for i in range(times):
            if rappor:
                max_sum, final_choice = self.rappor_baseline()
            else:
                max_sum, final_choice = self.perfect_baseline(trans_sample, baseline='noisy')
            choice_list.append(final_choice)
            true_noisy_party = true_sample[final_choice,:]
            fake_noisy_party = trans_sample[final_choice, :]
            sum_row = np.sum(fake_noisy_party, axis=0)
            real_sum = np.sum(np.sum(true_noisy_party, axis=0) > 0)
            noisy_result.append(real_sum)
            noisy_post.append(self.fast_post_sum(sum_row, fast_mode=True))



            rand_choice = random.sample(range(self.people), self.candidate_num)
            choice_list.append(rand_choice)

            true_rand_party = true_sample[rand_choice, :]
            fake_rand_party = trans_sample[rand_choice, :]

            # print out the posterior
            sum_row = np.sum(fake_rand_party, axis=0)
            # total_sum_rand = self.sum_posterior(sum_row)
            total_sum_rand = self.fast_post_sum(sum_row, fast_mode=True)
            coverage_rand = np.sum(np.sum(true_rand_party, axis=0) > 0)
            random_result.append(coverage_rand)
            random_post.append(total_sum_rand)

        result.append((np.mean(noisy_result), np.mean(noisy_post)))
        result.append((np.mean(random_result), np.mean(random_post)))

        # first element is no_privacy, second is our alg, third is fake-data, last is random
        percentile = np.array([r[0] for r in result])
        percentile = percentile / percentile[-1]
        # plot_by_loc(choice_list)
        return result, percentile

