import scipy.optimize.lbfgsb
import numpy as np
def posterior(p,eps,N):
    q = 1.0/((1.0-p)/(p*np.exp(eps)) + 1)
    upper_bound = 500
    p_0 = p
    p_1 = q

    xi = self.xi[plc_id]
    sum_ = 0
    numerator = p_0 ** X * (1 - p_0) ** (N - X)
    # use Gaussian to approximate Binomial when N is large
    if N > upper_bound:
        mu = N * p_0
        std = math.sqrt(N * p_0 * (1 - p_0))
        numerator *= Gaussian_Approx(X, mu, std)
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
                mu = (N - i) * p_0
                std = math.sqrt((N - i) * p_0 * (1 - p_0))
                second_part = Gaussian_Approx(X - m, mu, std)
            else:
                second_part = self.comb_mat[N - i, X - m] * p_0 ** (X - m) * (1 - p_0) ** (N - i - X + m)
            # inner += self.comb_mat[i, m] * p_1 ** m * (1 - p_1) ** (i - m) * \
            #         self.comb_mat[N - i, X - m] * p_0 ** (X - m) * (1 - p_0) ** (N - i - X + m)
            inner += first_part * second_part
        sum_ += outer * inner
    ratio = numerator / sum_
    return ratio