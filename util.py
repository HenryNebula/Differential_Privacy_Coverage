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


def plot_optim(name="log2.txt"):
    with open(name,'r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        lines = [l.split(' ') for l in lines]
    loss = np.array([float(l[0]) for l in lines])
    p = np.array([float(l[1]) for l in lines])
    index = np.argsort(p)
    p = p[index]
    loss = loss[index]
    plt.plot(p,loss)
    plt.title("Stepsize: 0.01")
    plt.ylabel("Loss: P(0|X)")
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

def plot_range():
    pass    
if __name__ == '__main__':
    # plot_epsilon()
    # plot_posterior()
    plot_optim()
    # cal_bound()
    # plot_range()