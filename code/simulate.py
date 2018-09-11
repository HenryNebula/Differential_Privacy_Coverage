from optim_coverage import *
import os
from itertools import product
from multiprocessing import Process
import time

def simulate_pipeline(candidate, k_favor, p, granu=0.02, plc_num=-1,loc=100, eps=4, hist=False):
    #todo: change optim_coverage parameter "hist" too
    people = -1
    candidate = candidate
    k_favor = k_favor
    p = p
    q = 1.0 / ((1.0 - p) / (p * np.exp(eps)) + 1)
    data_src = 'MCS'

    new_simulation = Diff_Coverage(flip_p=p, flip_q=q, candidate_num=candidate,
                                   plc_num=plc_num, people=people, k_favor=k_favor, max_iter=4000,
                                   data_src=data_src, uniform=False, random_start=5,
                                   obfuscate=0, granu=granu, hist=hist)

    if hist:
        data_src += '_hist_tmp'
    else:
        data_src += '_tmp'
    if 'MCS' in data_src:
        print "Data source:{0}, p:{1}, eps:{2}, candidate:{3}, k_favor:{4}".format(data_src, p, eps, candidate, k_favor)

        file_name = "".join([output_dir,data_src, "/k_{0}_cand_{1}_p_{2}_granu_{3}_eps_{4}.txt".format(
            k_favor, candidate, p, granu, eps)])

    else:
        print "Data source:{0}, p:{1}, eps:{2}, candidate:{3}, k_favor:{4}".format(data_src, p, eps, candidate, k_favor)

        file_name = "".join([output_dir, data_src, "/k_{0}_cand_{1}_p_{2:2.3f}_loc_{3}_eps_{4}.txt".format(
            k_favor, candidate, p, loc, eps)])

    if os.path.exists(file_name):
        print "File already exists for {}".format(file_name)
        return
    # time.sleep(np.random.random() * 60)

    true_sample, old_sample = new_simulation.real_sample()
    trans_sample = new_simulation.transfer_sample(true_sample)

    choice = new_simulation.train(trans_sample=trans_sample)
    result, percentile = new_simulation.validate(old_sample, trans_sample, choice, times=10)
    print percentile
    if not os.path.isdir(output_dir + data_src):
        os.mkdir(output_dir + data_src)

    with open(file_name, 'w') as f:
        f.writelines("People:{0}, Location:{1}\n".format(new_simulation.people, new_simulation.plc_num))
        f.writelines("Result:{}\n".format(result))
        f.writelines("Percentile:{}\n".format(percentile))


if __name__ == '__main__':

    data_src = 'SG'
    if data_src == 'SG':
        simulate_pipeline(1000, 5, p=0.02, granu=0.02, plc_num=-1, loc=100)
    else:
        cands = range(600,700,400)
        k_favor = [5]
        p = [0.02,1e-3]
        granu = [0.05]
        eps = range(1,11)

        paras = product(cands, k_favor, p, granu, eps)
        # paras = product(cands, k_favor, p, loc)
        pcs = []
        for parameter in paras:
            cand, k, p, granu, eps = parameter
            pcs.append(Process(target=simulate_pipeline, kwargs={
                "candidate":cand,
                "k_favor": k,
                "p": p,
                # "plc_num": loc
                "granu": granu,
                "eps": eps,
            }))
            pcs[-1].start()
        for process in pcs:
            process.join()