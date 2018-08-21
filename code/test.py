from multiprocessing import Pool,cpu_count, forking, Process
import concurrent.futures
import time
import os
from directory import *


def unwrap_self_f(arg, **kwarg):
    # print arg, kwarg
    return C.f(*arg, **kwarg)

def unwrap_self_run(arg, **kwarg):
    # print arg, kwarg
    return C.run(*arg, **kwarg)

class C:
    def f(self, name):
        with open(''.join([output_dir, name, '.txt']),'w') as f:
            f.write('hello %s,' % name)

    def run(self, num):
        pool = Pool(processes=2)
        print "num is {}".format(num)
        names = ('frank', 'justin', 'osi', 'thomas')
        pool.map(unwrap_self_f, zip([self] * len(names), names))

def starter(num):
    c = C()
    c.run(num)


if __name__ == '__main__':
    processes = [Process()] * 4
    for i in range(0,4):
        processes[i] = Process(target=starter,kwargs={'num':i})
        # processes[i].daemon = False
        processes[i].start()
    for i in range(0,4):
        processes[i].join()