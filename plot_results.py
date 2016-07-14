import numpy as np

from matplotlib.font_manager import findfont, FontProperties
import matplotlib.pyplot as plt
import sys

#for smoothing
kernel = np.array([1.] * 3)
kernel = kernel / np.sum(kernel)
# PATH = "/home/gabi/Documents/CSDL/deep_q_rl/pkl-results/results/"
# colors = ["red", "black", "#009900",  "#8c1aff", "#ff6600"]
# colors = ["red", "black", "#ff6600",  "#8c1aff", "#009900"]
colors = ["green", "blue", "red", "#00D3C6", "black"]
# lines = ["solid", "dashed", "dotted"]
lines = ["solid", ":"]
markers = ['o', "s", "^"]


def save_plot(name):
    plt.savefig(name+".png", format='png', dpi=400)
    pass


def load_csv(name):
    PATH = "/home/gabi/Documents/CSDL/deep_q_rl/pkl-results/learning/"
    return np.loadtxt(open(PATH+str(name)+".csv", "rb"), delimiter=",", skiprows=1) #np.ndarray

    pass

#each item in name_list is a np.ndarray containing floats, assume they are all same size
def getAvg(name_list):
    a = np.zeros(np.shape(name_list[0]))
    for i in range(len(name_list)):
        a = np.add(a, name_list[i])
    return a/len(name_list)
    pass


def getAvgClp(name_list, clip):
    a = np.zeros(np.shape(name_list[0][0:clip, :]))
    for i in range(len(name_list)):
        a = np.add(a, name_list[i][0:clip, :])
    return a / len(name_list)
    pass


def carnival_scatter(xcol, ycol):
    ca_nips = load_csv("carnival-nips")
    ca_da_1 = load_csv("carnival_da_2186039")
    ca_da_2 = load_csv("carnival_da_2215046")
    ca_da_3 = load_csv("carnival_da_2215048")
    ca_da = getAvg([ca_da_1, ca_da_2, ca_da_3])
    ca_si_1 = load_csv("carnival_si_2186032")
    ca_si_2 = load_csv("carnival_si_2215038")
    ca_si_3 = load_csv("carnival_si_2215041")
    ca_si = getAvg([ca_si_1, ca_si_2, ca_si_3])
    edgecol = "#0d0d0d"
    alp = 0.95
    plt.scatter(ca_nips[:, xcol], ca_nips[:, ycol], label="CA-nips", color=colors[0], linestyle=lines[0], marker=markers[0], s=50.0, alpha=alp, edgecolors=edgecol)
    plt.scatter(ca_da[:, xcol], ca_da[:, ycol], label="DA", color=colors[1], linestyle=lines[1], marker=markers[1], s=50.0, alpha=alp, edgecolors=edgecol)
    plt.scatter(ca_si[:, xcol], ca_si[:, ycol], label="SI", color=colors[2], linestyle=lines[2], marker=markers[2], s=50.0, alpha=alp, edgecolors=edgecol)
    labs = ["Number of epochs", "Number of Episodes", "Total Reward per Epoch", "Average Reward per Episode", "Average Action Value (Q)"]
    savename = ["num_epochs", "num_episode_carnival", "tot_rew_epoch_carnival", "avg_rew_episode_carnival", "avg_q_carnival"]
    plt.xlabel(labs[xcol])
    plt.ylabel(labs[ycol])
    plt.legend(loc="upper left")
    save_plot("scatter__"+str(savename[xcol]+"__"+str(savename[ycol])))
    plt.show()
    pass


def da_scatter(xcol, ycol):
    da_nips = load_csv("demon_attack-nips")
    da_ca_1 = load_csv("demon_attack_ca_2186052")
    da_ca_2 = load_csv("demon_attack_ca_2215081")
    da_ca_3 = load_csv("demon_attack_ca_2215086")
    da_ca = getAvg([da_ca_1, da_ca_2, da_ca_3])
    da_si_1 = load_csv("demon_attack_si_2186057")
    da_si_2 = load_csv("demon_attack_si_2215095")
    da_si_3 = load_csv("demon_attack_si_2215104")
    da_si = getAvg([da_si_1, da_si_2, da_si_3])
    edgecol = "#0d0d0d"
    alp = 0.95
    plt.scatter(da_nips[:, xcol], da_nips[:, ycol], label="CA-nips", color=colors[0], linestyle=lines[0],
                marker=markers[0], s=50.0, alpha=alp, edgecolors=edgecol)
    plt.scatter(da_ca[:, xcol], da_ca[:, ycol], label="DA", color=colors[1], linestyle=lines[1], marker=markers[1],
                s=50.0, alpha=alp, edgecolors=edgecol)
    plt.scatter(da_si[:, xcol], da_si[:, ycol], label="SI", color=colors[2], linestyle=lines[2], marker=markers[2],
                s=50.0, alpha=alp, edgecolors=edgecol)
    labs = ["Number of epochs", "Number of Episodes", "Total Reward per Epoch", "Average Reward per Episode", "Average Action Value (Q)"]
    savename = ["num_epochs", "num_episode_carnival", "tot_rew_epoch_carnival", "avg_rew_episode_carnival", "avg_q_carnival"]
    plt.xlabel(labs[xcol])
    plt.ylabel(labs[ycol])
    plt.legend(loc="lower right")
    save_plot("scatter__"+str(savename[xcol]+"__"+str(savename[ycol])))
    plt.show()
    pass


def si_scatter(xcol, ycol):
    si_nips = load_csv("space_invaders-nips")
    si_ca_1 = load_csv("space_invaders_ca_2186042")
    si_ca_2 = load_csv("space_invaders_ca_2215055")
    si_ca_3 = load_csv("space_invaders_ca_2215056")
    si_ca = getAvg([si_ca_1, si_ca_2, si_ca_3])
    si_da_1 = load_csv("space_invaders_da_2186047")
    si_da_2 = load_csv("space_invaders_da_2215058")
    si_da_3 = load_csv("space_invaders_da_2215059")
    si_da = getAvg([si_da_1, si_da_2, si_da_3])
    plt.scatter(si_nips[:, xcol], si_nips[:, ycol], label="SI-nips", color=colors[0], linestyle=lines[0])
    plt.scatter(si_ca[:, xcol], si_ca[:, ycol], label="CA", color=colors[1], linestyle=lines[1])
    plt.scatter(si_da[:, xcol], si_da[:, ycol], label="DA", color=colors[2], linestyle=lines[2])
    labs = ["Number of epochs", "Number of Episodes", "Total Reward per Epoch", "Average Reward per Episode", "Average Action Value (Q)"]
    savename = ["num_epochs", "num_episode_carnival", "tot_rew_epoch_carnival", "avg_rew_episode_carnival", "avg_q_carnival"]
    plt.xlabel(labs[xcol])
    plt.ylabel(labs[ycol])
    plt.legend(loc="best")
    save_plot("scatter__"+str(savename[xcol]+"__"+str(savename[ycol])))
    plt.show()
    pass


#plot everything
#epoch,num_episodes,total_reward,reward_per_episode,mean_q
def all_the_plots_carnival(col):
    ylab = ["Number of Episodes", "Total Reward per Epoch", "avg R/epi", "avg Q"]
    savename = ["num_episode_carnival", "tot_rew_epoch_carnival", "avg_rew_episode_carnival", "avg_q_carnival"]
    llw = 0.5
    plt.figure(figsize=(14, 3))
    # print(plt.rcParams.keys())

    axes = plt.gca()
    axes.set_ylim([0.5, 2.5])
    # axes.set_xlim([0, 6000])

    ca_nips = load_csv("carnival_nips_2237007")

    # 50 epoch
    ca_da_1_50 = load_csv("carnival_50_da_2243663")
    ca_da_2_50 = load_csv("carnival_50_da_2243665")
    ca_da_3_50 = load_csv("carnival_50_da_2243667")
    ca_da_50 = getAvg([ca_da_1_50, ca_da_2_50, ca_da_3_50])
    ca_si_1_50 = load_csv("carnival_50_si_2243463")
    ca_si_2_50 = load_csv("carnival_50_si_2243649")
    ca_si_3_50 = load_csv("carnival_50_si_2243653")
    ca_si_50 = getAvg([ca_si_1_50, ca_si_2_50, ca_si_3_50])
    # 100 epoch
    ca_da_1_100 = load_csv("carnival_100_da_2243669")
    ca_da_2_100 = load_csv("carnival_100_da_2243672")
    ca_da_3_100 = load_csv("carnival_100_da_2243676")
    ca_da_100 = getAvg([ca_da_1_100, ca_da_2_100, ca_da_3_100])
    ca_si_1_100 = load_csv("carnival_100_si_2243655")
    ca_si_2_100 = load_csv("carnival_100_si_2243656")
    ca_si_3_100 = load_csv("carnival_100_si_2243659")
    ca_si_100 = getAvg([ca_si_1_100, ca_si_2_100, ca_si_3_100])

    print(str(savename[col-1])+" ca original")
    print(ca_nips[-1, col])
    print(str(savename[col - 1]) + " da ca 50")
    print(ca_da_50[-1, col])
    print(str(savename[col - 1]) + " da ca 100")
    print(ca_da_100[-1, col])
    print(str(savename[col - 1]) + " si ca 50")
    print(ca_si_50[-1, col])
    print(str(savename[col - 1]) + " si ca 100")
    print(ca_si_100[-1, col])


    # plt.plot(ca_nips[:, 0], ca_nips[:, col], label="CA_s", color=colors[0], linestyle=lines[0], lw=llw+0.9)
    #
    # plt.plot(ca_da_50[:, 0], ca_da_50[:, col], label="DA 50", color=colors[1], linestyle=lines[0], lw=llw+0.9)
    # plt.plot(ca_da_100[:, 0], ca_da_100[:, col], label="DA 100", color=colors[1], linestyle=lines[1], lw=llw+2.5)
    #
    # plt.plot(ca_si_50[:, 0], ca_si_50[:, col], label="SI 50", color=colors[2], linestyle=lines[0], lw=llw+0.9)
    # plt.plot(ca_si_100[:, 0], ca_si_100[:, col], label="SI 100", color=colors[2], linestyle=lines[1], lw=llw+2.5)
    #
    # plt.ylabel(ylab[col-1])
    # # plt.xlabel("Training Epochs")
    # plt.title("Playing Carnival")
    # # plt.legend(bbox_to_anchor=(1.12, 1.035),borderpad=0.3, fontsize=12)
    # leg=plt.legend(loc=9, borderpad=0.3, fontsize=12, ncol=5)
    #
    # # plt.legend(loc="best")
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)
    # save_plot(savename[col-1]+"all")
    # plt.show()
    pass


#plot everything
#epoch,num_episodes,total_reward,reward_per_episode,mean_q
def all_the_plots_dem_at(col):
    ylab = ["Number of Episodes", "Total Reward per Epoch", "avg R/epi", "avg Q"]
    savename = ["num_episode_dem_at", "tot_rew_epoch_dem_at", "avg_rew_episode_dem_at", "avg_q_dem_at"]
    mme = 4
    llw = 0.5
    ss = 5


    plt.figure(figsize=(14, 3))
    axes = plt.gca()
    axes.set_ylim([0.5, 2.5])
    da_nips = load_csv("demon_attack_nips_2237014")
    #50 epochs
    da_ca_1_50 = load_csv("demon_attack_50_ca_2244113")
    da_ca_2_50 = load_csv("demon_attack_50_ca_2244116")
    da_ca_3_50 = load_csv("demon_attack_50_ca_2244119")
    da_ca_50 = getAvg([da_ca_1_50, da_ca_2_50, da_ca_3_50])
    da_si_1_50 = load_csv("demon_attack_50_si_2244128")
    da_si_2_50 = load_csv("demon_attack_50_si_2244129")
    da_si_3_50 = load_csv("demon_attack_50_si_2244131")
    da_si_50 = getAvg([da_si_1_50, da_si_2_50, da_si_3_50])
    #100 epochs
    da_ca_1_100 = load_csv("demon_attack_100_ca_2244123")
    da_ca_2_100 = load_csv("demon_attack_100_ca_2244125")
    da_ca_3_100 = load_csv("demon_attack_100_ca_2244126")
    da_ca_100 = getAvg([da_ca_1_100, da_ca_2_100, da_ca_3_100])
    da_si_1_100 = load_csv("demon_attack_100_si_2244201")
    da_si_2_100 = load_csv("demon_attack_100_si_2244204")
    da_si_3_100 = load_csv("demon_attack_100_si_2244206")
    da_si_100 = getAvg([da_si_1_100, da_si_2_100, da_si_3_100])

    print(str(savename[col-1])+" da original")
    print(da_nips[-1, col])
    print(str(savename[col - 1]) + " ca da 50")
    print(da_ca_50[-1, col])
    print(str(savename[col - 1]) + " ca da 100")
    print(da_ca_100[-1, col])
    print(str(savename[col - 1]) + " si da 50")
    print(da_si_50[-1, col])
    print(str(savename[col - 1]) + " si da 100")
    print(da_si_100[-1, col])


    # plt.plot(da_nips[:, 0], da_nips[:, col], label="DA_s", color=colors[0], linestyle=lines[0], lw=llw+0.9)
    #
    # plt.plot(da_ca_50[:, 0], da_ca_50[:, col], label="CA 50", color=colors[1], linestyle=lines[0], lw=llw+0.9)
    # plt.plot(da_ca_100[:, 0], da_ca_100[:, col], label="CA 100", color=colors[1], linestyle=lines[1], lw=llw+2.5)
    #
    # plt.plot(da_si_50[:, 0], da_si_50[:, col], label="SI 50", color=colors[2], linestyle=lines[0], lw=llw+0.9)
    # plt.plot(da_si_100[:, 0], da_si_100[:, col], label="SI 100", color=colors[2], linestyle=lines[1], lw=llw+2.5)
    #
    # plt.ylabel(ylab[col-1])
    # # plt.xlabel("Training Epochs")
    # plt.title("Playing Demon Attack")
    # # plt.legend(bbox_to_anchor=(1.12, 1.035),borderpad=0.3, fontsize=12)
    # leg = plt.legend(loc=9, borderpad=0.3, fontsize=12, ncol=5)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)
    # save_plot(savename[col-1]+"all")
    # plt.show()
    pass


#plot everything
#epoch,num_episodes,total_reward,reward_per_episode,mean_q
def all_the_plots_sp_in(col):
    ylab = ["Number of Episodes", "Total Reward per Epoch", "avg R/epi", "avg Q"]
    savename = ["num_episode_sp_in", "tot_rew_epoch_sp_in", "avg_rew_episode_sp_in", "avg_q_sp_in"]
    llw = 0.5
    plt.figure(figsize=(14, 3))
    si_nips = load_csv("space_invaders_nips_2237009")
    plt.subplot(111)
    axes = plt.gca()
    axes.set_ylim([0.5, 2.5])
    #50 epoch
    si_ca_1_50 = load_csv("space_invaders_50_ca_2243679")
    si_ca_2_50 = load_csv("space_invaders_50_ca_2243681")
    si_ca_3_50 = load_csv("space_invaders_50_ca_2243682")
    si_ca_50 = getAvg([si_ca_1_50, si_ca_2_50, si_ca_3_50])
    si_da_1_50 = load_csv("space_invaders_50_da_2244102")
    si_da_2_50 = load_csv("space_invaders_50_da_2244105")
    si_da_3_50 = load_csv("space_invaders_50_da_2244106")
    si_da_50 = getAvg([si_da_1_50, si_da_2_50, si_da_3_50])
    #100 epoch
    si_ca_1_100 = load_csv("space_invaders_100_ca_2243684")
    si_ca_2_100 = load_csv("space_invaders_100_ca_2243695")
    si_ca_3_100 = load_csv("space_invaders_100_ca_2244098")
    si_ca_100 = getAvg([si_ca_1_100, si_ca_2_100, si_ca_3_100])
    si_da_1_100 = load_csv("space_invaders_100_da_2244108")
    si_da_2_100 = load_csv("space_invaders_100_da_2244110")
    si_da_3_100 = load_csv("space_invaders_100_da_2244111")
    si_da_100 = getAvg([si_da_1_100, si_da_2_100, si_da_3_100])

    print(str(savename[col-1])+" si original")
    print(si_nips[-1, col])
    print(str(savename[col - 1]) + " ca si 50")
    print(si_ca_50[-1, col])
    print(str(savename[col - 1]) + " ca si 100")
    print(si_ca_100[-1, col])
    print(str(savename[col - 1]) + " da si 50")
    print(si_da_50[-1, col])
    print(str(savename[col - 1]) + " da si 100")
    print(si_da_100[-1, col])


    # plt.plot(si_nips[:, 0], si_nips[:, col], label="SI_s", color=colors[0], linestyle=lines[0], lw=llw+0.9)
    #
    # plt.plot(si_ca_50[:, 0], si_ca_50[:, col], label="CA 50", color=colors[1], linestyle=lines[0], lw=llw+0.9)
    # plt.plot(si_ca_100[:, 0], si_ca_100[:, col], label="CA 100", color=colors[1], linestyle=lines[1], lw=llw+2.5)
    #
    # plt.plot(si_da_50[:, 0], si_da_50[:, col], label="DA 50", color=colors[2], linestyle=lines[0], lw=llw+0.9)
    # plt.plot(si_da_100[:, 0], si_da_100[:, col], label="DA 100", color=colors[2], linestyle=lines[1], lw=llw+2.5)
    #
    # plt.ylabel(ylab[col-1])
    # plt.xlabel("Training Epochs")
    # plt.title("Playing Space Invaders")
    # # plt.legend(loc="best")
    # # plt.legend(bbox_to_anchor=(1.12, 1.035),borderpad=0.3, fontsize=12)
    # leg=plt.legend(loc=9, borderpad=0.3, fontsize=12, ncol=5)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)
    # save_plot(savename[col-1]+"all")
    # plt.show()
    pass


def plot_loss_ca():
    llw = 1
    plt.figure(figsize=(14, 3))
    clip = 9900
    kernel = np.array([1.] * 8)
    kernel = kernel / np.sum(kernel)

    ca_nips = load_csv("carnival_nips_2237007")
    # 50 epoch
    ca_da_1_50 = load_csv("carnival_50_da_2243663")
    ca_da_2_50 = load_csv("carnival_50_da_2243665")
    ca_da_3_50 = load_csv("carnival_50_da_2243667")
    ca_da_50 = getAvgClp([ca_da_1_50, ca_da_2_50, ca_da_3_50], clip)
    ca_si_1_50 = load_csv("carnival_50_si_2243463")
    ca_si_2_50 = load_csv("carnival_50_si_2243649")
    ca_si_3_50 = load_csv("carnival_50_si_2243653")
    ca_si_50 = getAvgClp([ca_si_1_50, ca_si_2_50, ca_si_3_50], clip)
    # 100 epoch
    ca_da_1_100 = load_csv("carnival_100_da_2243669")
    ca_da_2_100 = load_csv("carnival_100_da_2243672")
    ca_da_3_100 = load_csv("carnival_100_da_2243676")
    ca_da_100 = getAvgClp([ca_da_1_100, ca_da_2_100, ca_da_3_100], clip)
    ca_si_1_100 = load_csv("carnival_100_si_2243655")
    ca_si_2_100 = load_csv("carnival_100_si_2243656")
    ca_si_3_100 = load_csv("carnival_100_si_2243659")
    ca_si_100 = getAvgClp([ca_si_1_100, ca_si_2_100, ca_si_3_100], clip)
    axes = plt.gca()
    axes.set_ylim([0.07, 0.17])
    axes.set_xlim([0,9800])
    print("loss ca original")
    print(ca_nips[-1, 0])
    print("loss da ca 50")
    print(ca_da_50[-1, 0])
    print("loss da ca 100")
    print(ca_da_100[-1, 0])
    print("loss si ca 50")
    print(ca_si_50[-1, 0])
    print("loss si ca 100")
    print(ca_si_100[-1, 0])

    # li_ca_nips = ca0_da_50[0:clip, 0].tolist()
    # print(max(li_ca_nips))
    # print(max(ca_nips[0:clip, 0].tolist()))
    # print(max(ca_da_50[0:clip, 0].tolist()))
    # print(max(ca_da_100[0:clip, 0].tolist()))
    # print(max(ca_si_50[0:clip, 0].tolist()))
    # print(max(ca_si_100[0:clip, 0].tolist()))

    # colors = ["green", "blue", "#00D3C6", "black", "grey"]
    #
    # plt.plot(np.convolve(ca_nips[0:clip, 0], kernel, mode="same"), label="CA_s", color=colors[0], linestyle=lines[0], lw=0.9)
    #
    # plt.plot(np.convolve(ca_da_50[:, 0], kernel, mode="same"), label="DA 50", color=colors[1], linestyle=lines[0], lw=0.9)
    # plt.plot(np.convolve(ca_da_100[:, 0], kernel, mode="same"), label="DA 100", color=colors[2], linestyle=lines[0], lw=0.9)
    #
    # plt.plot(np.convolve(ca_si_50[:, 0], kernel, mode="same"), label="SI 50", color=colors[3], linestyle=lines[0], lw=0.9)
    # plt.plot(np.convolve(ca_si_100[:, 0], kernel, mode="same"), label="SI 100", color=colors[4], linestyle=lines[0], lw=0.9)
    #
    # plt.ylabel("Avg Loss")
    # plt.xlabel("Total Updates")
    # plt.title("Loss Carnival")
    # # plt.legend(loc="best")
    # leg = plt.legend(loc=8, borderpad=0.3, fontsize=12, ncol=5)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)
    # save_plot("carnival_loss")
    # plt.show()

    pass

def plot_loss_da():
    ylab = ["Number of Episodes", "Total Reward per Epoch", "Average Reward per Episode", "Average Action Value (Q)"]
    savename = ["num_episode_carnival", "tot_rew_epoch_carnival", "avg_rew_episode_carnival", "avg_q_carnival"]
    clip = 6000
    kernel = np.array([1.] * 3)
    kernel = kernel / np.sum(kernel)

    plt.figure(figsize=(14, 3))
    da_nips = load_csv("demon_attack_nips_2237014")
    # 50 epochs
    da_ca_1_50 = load_csv("demon_attack_50_ca_2244113")
    da_ca_2_50 = load_csv("demon_attack_50_ca_2244116")
    da_ca_3_50 = load_csv("demon_attack_50_ca_2244119")
    da_ca_50 = getAvgClp([da_ca_1_50, da_ca_2_50, da_ca_3_50], clip)
    da_si_1_50 = load_csv("demon_attack_50_si_2244128")
    da_si_2_50 = load_csv("demon_attack_50_si_2244129")
    da_si_3_50 = load_csv("demon_attack_50_si_2244131")
    da_si_50 = getAvgClp([da_si_1_50, da_si_2_50, da_si_3_50], clip)
    # 100 epochs
    da_ca_1_100 = load_csv("demon_attack_100_ca_2244123")
    da_ca_2_100 = load_csv("demon_attack_100_ca_2244125")
    da_ca_3_100 = load_csv("demon_attack_100_ca_2244126")
    da_ca_100 = getAvgClp([da_ca_1_100, da_ca_2_100, da_ca_3_100], clip)
    da_si_1_100 = load_csv("demon_attack_100_si_2244201")
    da_si_2_100 = load_csv("demon_attack_100_si_2244204")
    da_si_3_100 = load_csv("demon_attack_100_si_2244206")
    da_si_100 = getAvgClp([da_si_1_100, da_si_2_100, da_si_3_100], clip)

    axes = plt.gca()
    axes.set_ylim([0.07, 0.13])
    axes.set_xlim([0,6000])
    print("loss da original")
    print(da_nips[-1, 0])
    print("loss ca da 50")
    print(da_ca_50[-1, 0])
    print("loss ca da 100")
    print(da_ca_100[-1, 0])
    print("loss si da 50")
    print(da_si_50[-1, 0])
    print("loss si a 100")
    print(da_si_100[-1, 0])

    # colors = ["green", "blue", "#00D3C6", "black", "grey"]
    #
    # plt.plot(np.convolve(da_nips[0:clip, 0], kernel, mode="same"), label="DA_s", color=colors[0], linestyle=lines[0], lw=0.9)
    #
    # plt.plot(np.convolve(da_ca_50[:, 0], kernel, mode="same"), label="CA 50", color=colors[1], linestyle=lines[0], lw=0.9)
    # plt.plot(np.convolve(da_ca_100[:, 0], kernel, mode="same"), label="CA 100", color=colors[2], linestyle=lines[0], lw=0.9)
    #
    # plt.plot(np.convolve(da_si_50[:, 0], kernel, mode="same"), label="SI 50", color=colors[3], linestyle=lines[0], lw=0.9)
    # plt.plot(np.convolve(da_si_100[:, 0], kernel, mode="same"), label="SI 100", color=colors[4], linestyle=lines[0], lw=0.9)
    #
    # plt.ylabel("Avg Loss")
    # plt.xlabel("Total Updates")
    # plt.title("Loss Demon Attack")
    # leg = plt.legend(loc=8, borderpad=0.3, fontsize=12, ncol=5)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)
    # save_plot("demonattack_loss")
    # plt.show()

    pass

def plot_loss_si():

    ylab = ["Number of Episodes", "Total Reward per Epoch", "Average Reward per Episode", "Average Action Value (Q)"]
    savename = ["num_episode_carnival", "tot_rew_epoch_carnival", "avg_rew_episode_carnival", "avg_q_carnival"]
    clip = 7500
    kernel = np.array([1.] * 3)
    kernel = kernel / np.sum(kernel)

    plt.figure(figsize=(14, 3))
    si_nips = load_csv("space_invaders_nips_2237009")
    #50 epoch
    si_ca_1_50 = load_csv("space_invaders_50_ca_2243679")
    si_ca_2_50 = load_csv("space_invaders_50_ca_2243681")
    si_ca_3_50 = load_csv("space_invaders_50_ca_2243682")
    si_ca_50 = getAvgClp([si_ca_1_50, si_ca_2_50, si_ca_3_50], clip)
    si_da_1_50 = load_csv("space_invaders_50_da_2244102")
    si_da_2_50 = load_csv("space_invaders_50_da_2244105")
    si_da_3_50 = load_csv("space_invaders_50_da_2244106")
    si_da_50 = getAvgClp([si_da_1_50, si_da_2_50, si_da_3_50], clip)
    #100 epoch
    si_ca_1_100 = load_csv("space_invaders_100_ca_2243684")
    si_ca_2_100 = load_csv("space_invaders_100_ca_2243695")
    si_ca_3_100 = load_csv("space_invaders_100_ca_2244098")
    si_ca_100 = getAvgClp([si_ca_1_100, si_ca_2_100, si_ca_3_100], clip)
    si_da_1_100 = load_csv("space_invaders_100_da_2244108")
    si_da_2_100 = load_csv("space_invaders_100_da_2244110")
    si_da_3_100 = load_csv("space_invaders_100_da_2244111")
    si_da_100 = getAvgClp([si_da_1_100, si_da_2_100, si_da_3_100], clip)

    print("loss si original")
    print(si_nips[-1, 0])
    print("loss ca si 50")
    print(si_ca_50[-1, 0])
    print("loss ca si 100")
    print(si_ca_100[-1, 0])
    print("loss da si 50")
    print(si_da_50[-1, 0])
    print("loss da si 100")
    print(si_da_100[-1, 0])

    # axes = plt.gca()
    # axes.set_ylim([0.07, 0.15])
    # axes.set_xlim([0, 7480])
    #
    # colors = ["green", "blue", "#00D3C6", "black", "grey"]
    #
    # plt.plot(np.convolve(si_nips[0:clip, 0], kernel, mode="same"), label="SI_s", color=colors[0], linestyle=lines[0], lw=0.9)
    #
    # plt.plot(np.convolve(si_ca_50[:, 0], kernel, mode="same"), label="CA 50", color=colors[1], linestyle=lines[0], lw=0.9)
    # plt.plot(np.convolve(si_ca_100[:, 0], kernel, mode="same"), label="CA 100", color=colors[2], linestyle=lines[0], lw=0.9)
    #
    # plt.plot(np.convolve(si_da_50[:, 0], kernel, mode="same"), label="DA 50", color=colors[3], linestyle=lines[0], lw=0.9)
    # plt.plot(np.convolve(si_da_100[:, 0], kernel, mode="same"), label="DA 100", color=colors[4], linestyle=lines[0], lw=0.9)
    #
    # plt.ylabel("Avg Loss")
    # plt.xlabel("Total Updates")
    # plt.title("Loss Space Invaders")
    # leg = plt.legend(loc=8, borderpad=0.3, fontsize=12, ncol=5)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)
    # save_plot("spaceinvaders_loss")
    # plt.show()

    pass

def reli((f_eps, f_loss)):
    eps = sum(f_eps[:, 1].tolist())
    loss_steps = len(f_loss[:, 0])
    reli = loss_steps / eps
    print("loss steps: " + str(loss_steps) + " eps: " + str(eps) + " reli: " + str(reli))
    return loss_steps, int(eps)
    pass

def loadstuff(name):
    PATH = "/home/gabi/Documents/CSDL/deep_q_rl/pkl-results/results/"
    f_eps = np.loadtxt(open(PATH + str(name) + ".csv", "rb"), delimiter=",", skiprows=1)
    PATH = "/home/gabi/Documents/CSDL/deep_q_rl/pkl-results/learning/"
    f_loss = np.loadtxt(open(PATH + str(name) + ".csv", "rb"), delimiter=",", skiprows=1)
    return f_eps, f_loss
    pass

def reli_loss():
    all = []
    #carnival
    all.append(reli(loadstuff("carnival_nips_2237007")))
    all.append(reli(loadstuff("carnival_50_da_2243663")))
    all.append(reli(loadstuff("carnival_50_da_2243665")))
    all.append(reli(loadstuff("carnival_50_da_2243667")))

    all.append(reli(loadstuff("carnival_50_si_2243463")))
    all.append(reli(loadstuff("carnival_50_si_2243649")))
    all.append(reli(loadstuff("carnival_50_si_2243653")))

    all.append(reli(loadstuff("carnival_100_da_2243669")))
    all.append(reli(loadstuff("carnival_100_da_2243672")))
    all.append(reli(loadstuff("carnival_100_da_2243676")))

    all.append(reli(loadstuff("carnival_100_si_2243655")))
    all.append(reli(loadstuff("carnival_100_si_2243656")))
    all.append(reli(loadstuff("carnival_100_si_2243659")))

    #demon attack
    all.append(reli(loadstuff("demon_attack_nips_2237014")))

    all.append(reli(loadstuff("demon_attack_50_ca_2244113")))
    all.append(reli(loadstuff("demon_attack_50_ca_2244116")))
    all.append(reli(loadstuff("demon_attack_50_ca_2244119")))

    all.append(reli(loadstuff("demon_attack_50_si_2244128")))
    all.append(reli(loadstuff("demon_attack_50_si_2244129")))
    all.append(reli(loadstuff("demon_attack_50_si_2244131")))

    all.append(reli(loadstuff("demon_attack_100_ca_2244123")))
    all.append(reli(loadstuff("demon_attack_100_ca_2244125")))
    all.append(reli(loadstuff("demon_attack_100_ca_2244126")))

    all.append(reli(loadstuff("demon_attack_100_si_2244201")))
    all.append(reli(loadstuff("demon_attack_100_si_2244204")))
    all.append(reli(loadstuff("demon_attack_100_si_2244206")))

    # space invaders
    all.append(reli(loadstuff("space_invaders_nips_2237009")))

    all.append(reli(loadstuff("space_invaders_50_ca_2243679")))
    all.append(reli(loadstuff("space_invaders_50_ca_2243681")))
    all.append(reli(loadstuff("space_invaders_50_ca_2243682")))

    all.append(reli(loadstuff("space_invaders_50_da_2244102")))
    all.append(reli(loadstuff("space_invaders_50_da_2244105")))
    all.append(reli(loadstuff("space_invaders_50_da_2244106")))

    all.append(reli(loadstuff("space_invaders_100_ca_2243684")))
    all.append(reli(loadstuff("space_invaders_100_ca_2243695")))
    all.append(reli(loadstuff("space_invaders_100_ca_2244098")))

    all.append(reli(loadstuff("space_invaders_100_da_2244108")))
    all.append(reli(loadstuff("space_invaders_100_da_2244110")))
    all.append(reli(loadstuff("space_invaders_100_da_2244111")))

    plt.figure(1)
    e = []
    l = []

    for i in xrange(len(all)):
        e.append(all[i][0])
        l.append(all[i][1])

    plt.scatter(e, l)
    plt.show()
    pass

#piemel
def plot_rms():
    layer1 = [[0.0, 0.330606, 0.400901, 0.436492, 0.441168, 0.496413],
              [0.330606, 0.0, 0.582233, 0.60643, 0.615712, 0.656092],
              [0.400901, 0.582233, 0.0, 0.101195, 0.369546, 0.430823],
              [0.436492, 0.60643, 0.101195, 0.0, 0.401304, 0.455846],
              [0.441168, 0.615712, 0.369546, 0.401304, 0.0, 0.140522],
              [0.496413, 0.656092, 0.430823, 0.455846, 0.140522, 0.0]]

    layer2 = [[0.0, 0.355273, 0.3455, 0.394582, 0.353221, 0.402024],
              [0.355273, 0.0, 0.546452, 0.579879, 0.553054, 0.586478],
              [0.3455, 0.546452, 0.0, 0.113403, 0.346499, 0.397164],
              [0.394582, 0.579879, 0.113403, 0.0, 0.395253, 0.440462],
              [0.353221, 0.553054, 0.346499, 0.395253, 0.0, 0.136527],
              [0.402024, 0.586478, 0.397164, 0.440462, 0.136527, 0.0]]

    layer3 = [[0.0, 0.0654225, 0.0778279, 0.0936997, 0.0816403, 0.0956956],
              [0.0654225, 0.0, 0.0966784, 0.10966, 0.0996592, 0.111318],
              [0.0778279, 0.0966784, 0.0, 0.0595404, 0.0785051, 0.0931361],
              [0.0936997, 0.10966, 0.0595404, 0.0, 0.0939481, 0.10639],
              [0.0816403, 0.0996592, 0.0785051, 0.0939481, 0.0, 0.0591807],
              [0.0956956, 0.111318,0.0931361, 0.10639, 0.0591807, 0.0]]

    layer4 = [[0.0, 0.106357, 0.277547, 0.295211, 0.208427, 0.220762],
              [0.106357, 0.0, 0.276403, 0.294355, 0.211666, 0.225816],
              [0.277547, 0.276403, 0.0, 0.0934813, 0.27011, 0.285578],
              [0.295211, 0.294355, 0.0934813, 0.0, 0.286666, 0.300777],
              [0.208427, 0.211666, 0.27011, 0.286666, 0.0, 0.0933578],
              [0.220762, 0.225816, 0.285578, 0.300777, 0.0933578, 0.0]]

    labels = ["CA 50", "CA 100", "DA 50", "DA 100", "SI 50", "SI 100"]

    fig = plt.figure(figsize=(9, 5))

    ax1 = fig.add_subplot(121)
    im = plt.imshow(layer3, vmin=0, vmax=0.31, interpolation="none",
           cmap="jet", aspect=1)
    plt.xlabel("FC 1")
    plt.yticks([0,1,2,3,4,5],labels)
    plt.tick_params(axis='x', labeltop="on", labelbottom="off")
    plt.xticks([0,1,2,3,4,5],labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.15)

    # cbar_ax = fig.add_axes([0.42, 0.285, 0.02, 0.485])
    # plt.colorbar(im, cax=cbar_ax)

    ax2 = fig.add_subplot(122)
    im2=plt.imshow(layer4, vmin=0, vmax=0.31, interpolation="none",
                    cmap="jet", aspect=1)
    plt.xlabel("FC 2")
    plt.yticks([])
    plt.tick_params(axis='x', labeltop="on", labelbottom="off")
    plt.xticks([0, 1, 2, 3, 4, 5], labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.15)

    fig.suptitle("RMS distance ", fontsize=14)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.285, 0.02, 0.485])
    plt.colorbar(im2, cax=cbar_ax)
    plt.show()
    fig.savefig("RMS_3_4" + ".png", format='png', dpi=400)
    pass


if __name__ == "__main__":
    # font = findfont(FontProperties(family=['sans-serif']))
    # print(font)
    # all_the_plots_carnival(3)
    # all_the_plots_carnival(4)
    # all_the_plots_dem_at(3)
    # all_the_plots_dem_at(4)
    # all_the_plots_sp_in(3)
    # all_the_plots_sp_in(4)
    # plot_loss_ca()
    # plot_loss_da()
    # plot_loss_si()
    # plt.show()
    # reli_loss()
    plot_rms()
    pass

