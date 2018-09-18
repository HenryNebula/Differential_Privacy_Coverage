from optim_coverage import *
import os
from itertools import product
from multiprocessing import Process
import time


def simulate_pipeline(candidate, k_favor, p, constraint, data_src, eps=4, hist=False, max_num=1):
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
    result, percentile = new_simulation.validate(choice, times=1)

    for i in range(max_num):
        file_name += '_'
        file_name += str(i)
        print percentile
        print 'Result:{}'.format(result)
        if not os.path.isdir(output_dir + data_src):
            os.mkdir(output_dir + data_src)

        with open(file_name, 'w') as f:
            f.writelines("People:{0}, Target:{1}, Non-Target:{2}\n".format(new_simulation.people, new_simulation.target_num,
                                                                           new_simulation.nnz_target_num))
            f.writelines("Result:{}\n".format(result))
            f.writelines("Percentile:{}\n".format(percentile))

        if max_num > 1:
            choice = new_simulation.train(optimizer='gradient', freeze=True)
            result, percentile = new_simulation.validate(choice, times=1)


if __name__ == '__main__':

    data_src = 'SG'
    if data_src == 'SG':
        cands = [40]
        k_favor = [7]
        p = [8e-2]
        eps = [4]
        granu = [250]
        hist = [False]

    else:
        cands = [600]
        k_favor = [5]
        plc_num = [1000]
        p = [0.02]
        eps = [4]
        granu = [0.02]
        hist = [False]

    paras = product(cands,k_favor, p, granu, eps, hist)
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
            "data_src":data_src
        }))
        pcs[-1].start()
    for process in pcs:
        process.join()