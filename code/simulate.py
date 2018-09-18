from optim_coverage import *
import os
from itertools import product
from multiprocessing import Process
import time


def simulate_pipeline(candidate, k_favor, p, constraint, data_src, eps=4, hist=False):
    candidate = candidate
    k_favor = k_favor
    p = p
    q = 1.0 / ((1.0 - p) / (p * np.exp(eps)) + 1)
    if q < 0.8:
        print("q:{0}, q is too small!".format(q))
        exit(1)
    elif eps > 10 and p > (1 - q) * 1e6:
        print("you can make p smaller, now p:{}, q:{}".format(p, q))
        exit(1)
    else:
        print(p,q, np.log((q*(1-p))/(p*(1-q))))

    new_simulation = Diff_Coverage(flip_p=p, flip_q=q, data_src=data_src,
                                   target_constraint=constraint, candidate_num=candidate,
                                   k_favor=k_favor, hist=hist, random_start=1)

    if hist:
        data_src += '_hist_tmp'
    else:
        data_src += '_tmp1'

    print "Data source:{0}, p:{1}, eps:{2}, candidate:{3}, k_favor:{4}".format(data_src, p, eps, candidate, k_favor)

    file_name = "".join([output_dir,data_src, "/k_{0}_cand_{1}_p_{2:2.3f}_constraint_{3}_eps_{4}.txt".format(
        k_favor, candidate, p, constraint, eps)])

    if os.path.exists(file_name):
        print "File already exists for {}".format(file_name)
        return

    choice = new_simulation.train(optimizer='gradient')
    new_simulation.rappor_baseline()
    result, percentile = new_simulation.validate(choice, times=5)
    print percentile
    print 'Result:{}'.format(result)
    if not os.path.isdir(output_dir + data_src):
        os.mkdir(output_dir + data_src)

    with open(file_name, 'w') as f:
        f.writelines("People:{0}, Target:{1}, Non-Target:{2}\n".format(new_simulation.people, new_simulation.target_num,
                                                                       new_simulation.nnz_target_num))
        f.writelines("Result:{}\n".format(result))
        f.writelines("Percentile:{}\n".format(percentile))


if __name__ == '__main__':

    data_src = 'SG'
    if data_src == 'SG':
<<<<<<< Updated upstream
        cands = [600]
        plc_num = [1500]
        p = [0.02]
        eps = range(1,15)
        granu = [0.02]
=======
        cands = range(300, 1100, 1200)
        plc_num = range(1000, 1200, 500)
        p = [5e-3]
        eps = [10]
>>>>>>> Stashed changes
        k_favor = [5]
        hist = [False]

        paras = product(cands, k_favor,p, plc_num, eps, hist)
    else:
        cands = [320]
        k_favor = [5]
<<<<<<< Updated upstream
        p = [0.02,1e-3]
        granu = [0.05]
        eps = range(1,15)
        hist = [True]
        plc_num = [-1]
=======
        p = [0.0001]
        granu = [0.035]
        eps = [40]
        hist = [True, False]
>>>>>>> Stashed changes

        paras = product(cands, k_favor, p, granu, eps, hist)
        # paras = product(cands, k_favor, p, loc)
    pcs = []
    for parameter in paras:
        cand, k, p, granu, eps, hist = parameter
        pcs.append(Process(target=simulate_pipeline, kwargs={
            "candidate":cand,
            "k_favor": k,
            "p": p,
            "constraint": granu,
            "eps": eps,
            "hist": hist,
<<<<<<< Updated upstream
            "data_src":data_src
=======
            "data_src": data_src
>>>>>>> Stashed changes
        }))
        pcs[-1].start()
    for process in pcs:
        process.join()