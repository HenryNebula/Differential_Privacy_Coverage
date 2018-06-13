from multiprocessing import Pool
import time


def unwrap_self_f(arg, **kwarg):
    print arg, kwarg
    return C.f(*arg, **kwarg)


class C:
    def f(self, name):
        print 'hello %s,' % name
        print 'nice to meet you.'

    def run(self):
        pool = Pool(processes=2)
        names = ('frank', 'justin', 'osi', 'thomas')
        pool.map(unwrap_self_f, zip([self] * len(names), names))


if __name__ == '__main__':
    c = C()
    c.run()
