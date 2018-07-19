from diff_coverage import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_epsilon():
    def f(p,q):
            return np.log(q*(1-p)/p/(1-q))
    # p: 0->1
    p_range = np.arange(0.001,0.5,0.01)
    # q: 1->1
    q_range = np.arange(0.501,1,0.01)

    X, Y = np.meshgrid(p_range, q_range)
    num = 10
    ax = plt.subplot(111)
    plt.contourf(X, Y, f(X,Y), num, alpha=1)
    C = plt.contour(X, Y, f(X,Y), num, colors='black', linewidth=0.5)
    plt.clabel(C, inline=True, fontsize=10)
    plt.xlabel("p")
    plt.ylabel("q")
    plt.title("$\epsilon(p,q) = \\log(\\frac{q(1-p)}{p(1-q)})$")
    xminorLocator = MultipleLocator(0.01)
    yminorLocator = MultipleLocator(0.01)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # plt.xticks(p_range)
    plt.show()


def plot_posterior():
    # p: 0->1
    p_range = np.arange(0.001,0.5,0.01)
    # q: 1->1
    q_range = np.arange(0.501,1,0.01)
    plc_num = 1000
    people = 600
    candidate = 100
    p = 0.02
    q = 0.6
    k_favor = 30
    skew = False

    obj = Diff_Coverage(flip_p=p, flip_q=q, candidate_num=candidate,
                                   plc_num=plc_num, people=people, k_favor=k_favor, max_iter=4000,
                                   data_src="MCS", uniform=False, freeze=False, skew=skew, random_start=1)
    obj.train(use_grad=True)
    X_u=range(0,9)
    def f(p,q,x):
        obj.p, obj.q = p, q
        xi = obj.xi
        print xi[0]
        prob_ = obj.posterior(x, 0)
        return prob_
    X, Y = np.meshgrid(p_range, q_range)
    num = 8
    plt.figure(figsize=(10.24,7.80))
    for i, x in enumerate(X_u):
        ax = plt.subplot(3,3,i+1)
        plt.contourf(X, Y, f(X, Y,x), num, alpha=1)
        C = plt.contour(X, Y, f(X, Y,x), num, colors='black', linewidth=0.5)
        plt.clabel(C, inline=True, fontsize=10)
        plt.xlabel('p')
        plt.ylabel('q')
        plt.title('X='+str(x))
        xminorLocator = MultipleLocator(0.01)
        yminorLocator = MultipleLocator(0.01)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
    # plt.xticks(p_range)
    plt.show()


def plot_optim(name="log.txt"):
    with open(name,'r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        lines = [l.split(' ') for l in lines]
    loss = np.array([float(l[0]) for l in lines])
    p = np.array([float(l[1]) for l in lines])
    index = np.argsort(p)
    p = p[index]
    loss = loss[index]
    plt.plot(p,900-loss)
    plt.title("Stepsize: 0.01")
    plt.ylabel("Posterior")
    plt.xlabel("p:Prob(0->1)")
    plt.show()

def cal_bound(name='new_log.txt'):
    with open(name, 'r') as f:
        lines = f.readlines()
        lines = [eval(l) for l in lines if '#' not in l]
    for l in lines:
        arr = [tup[0] for tup in l]
        arr = (np.array(arr) - arr[3]) / (arr[3]+0.0)*100
        print arr


def easy_draw(d=None,fpath='change_prior/'):
    if d is None:
        with open(fpath + "simu.json", 'r') as f:
            d = json.loads(f.read())
    dataset_num = d['dataset_num']
    method_num = d['method_num']
    # method_num = 3
    iter_per_set = d['iter_per_set']
    dir_name = d['dir_name']
    data_src = d['data_src']
    marker = d['change_paras']
    # title = d['title']

    #     marker = ['cand=30','cand=60','cand=90','cand=120']
    #     marker = ['k=10','k=20','k=30']
    #     marker  ['cand=40','cand=42','cand=44']

    color = ['y', 'c', 'b', 'r', 'm', 'k']
    Results = np.zeros((dataset_num, method_num, iter_per_set))
    Stats = np.zeros((dataset_num, method_num, 2))  # 0 for mean, 1 for std

    for i in range(0, dataset_num):
        for j in range(0, iter_per_set):
            with open(dir_name + '/' + data_src + '_' +
                      str(i) + '_' + str(j) + ".json", 'r') as f:
                results = json.loads(f.read())
                for m in range(0, method_num):
                    Results[i, m, j] = results[m][0]

    for i in range(0, dataset_num):
        for m in range(0, method_num):
            Stats[i, m, 0] = np.mean(Results[i, m, :])
            Stats[i, m, 1] = np.std(Results[i, m, :])

    print Stats[0, 0, 1]
    ind = np.arange(dataset_num)
    width = 0.2
    p_list = []
    for m in range(0, method_num):
        if method_num % 2 != 0:
            bias = np.floor(method_num / 2)
        else:
            bias = (method_num - 1.0) / 2
        p = plt.bar(ind + width * (m - bias), Stats[:, m, 0], width=width, color=color[m], yerr=Stats[:, m, 1])
        p_list.append(p)

    plt.xticks(ind, marker)
    # plt.ylim(500, 650)
    tup = tuple([p[0] for p in p_list])
    plt.legend(tup, ("No Noise","Greedy", "Noisy", "Random"))
    plt.ylabel("Num of Covered Targets")
    # if 'title' in d.keys():
    #     plt.title(d['title'])
    # else:
    #     plt.title(d['data_src'])
    plt.savefig(fpath + d['pic_name'])
    plt.show()

if __name__ == '__main__':
    # plot_epsilon()
    # plot_posterior()
    # plot_optim()
    # cal_bound()
    # plot_range()
    easy_draw(fpath='test/')