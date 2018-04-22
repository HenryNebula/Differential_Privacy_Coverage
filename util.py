from diff_coverage import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_epsilon():
    def f(p,q):
            return np.log(q*(1-p)/p/(1-q))
    # p: 0->1
    p_range = np.arange(0.001,0.5,0.01)
    # q: 1->1
    q_range = np.arange(0.501,1,0.01)
    # epsilon = np.zeros((p_range.shape[0], q_range.shape[0]))
    # for row, p in enumerate(p_range):
    #     for col, q in enumerate(q_range):
    #         epsilon[row, col] = np.log(q*(1-p)/p/(1-q))
    # plt.imshow(epsilon,extent=[0,0.5,1,0.5],interpolation="none", aspect="auto")
    # x = np.arange(0,0.5,0.01)
    # y = 1 - x
    # plt.plot(x,y)

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
                                   data_src="SG", uniform=False, freeze=False, skew=skew)
    obj.train(use_grad=True)
    X_u=[30,60,90]
    def f(p,q,x):
        obj.p, obj.q = p, q
        xi = obj.xi
        print xi[54]
        prob_ = obj.posterior(x, 54)
        return prob_
    X, Y = np.meshgrid(p_range, q_range)
    num = 8
    for i, x in enumerate(X_u):
        ax = plt.subplot(len(X_u),1,i+1)
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




if __name__ == '__main__':
    plot_epsilon()
    # plot_posterior()