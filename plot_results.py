import numpy as np
import matplotlib.pyplot as plt
import sys

#for smoothing
kernel = np.array([1.] * 5)
kernel = kernel / np.sum(kernel)
PATH = "/home/gabi/Documents/CSDL/deep_q_rl/pkl-results/results/"
colors = ["#00cc00", "#ff0066", "#336699"]
# lines = ["solid", "dashed", "dotted"]
lines = ["solid", "solid", "solid"]



def save_plot(name):
    plt.savefig(name+".png", format='png', dpi=400)
    pass


def load_csv(name):
    return np.loadtxt(open(PATH+str(name)+".csv", "rb"), delimiter=",", skiprows=1) #np.ndarray
    pass


#each item in name_list is a np.ndarray containing floats, assume they are all same size
def getAvg(name_list):
    a = np.zeros(np.shape(name_list[0]))
    for i in range(len(name_list)):
        a = np.add(a, name_list[i])
    return a/len(name_list)
    pass


#plot everything
#epoch,num_episodes,total_reward,reward_per_episode,mean_q
def all_the_plots_carnival(col):
    ylab = ["Number of Episodes", "Total Reward per Epoch", "Average Reward per Episode", "Average Action Value (Q)"]
    savename = ["num_episode_carnival", "tot_rew_epoch_carnival", "avg_rew_episode_carnival", "avg_q_carnival"]

    ca_nips = load_csv("carnival-nips")
    ca_da_1 = load_csv("carnival_da_2186039")
    ca_da_2 = load_csv("carnival_da_2215046")
    ca_da_3 = load_csv("carnival_da_2215048")
    ca_da = getAvg([ca_da_1, ca_da_2, ca_da_3])
    ca_si_1 = load_csv("carnival_si_2186032")
    ca_si_2 = load_csv("carnival_si_2215038")
    ca_si_3 = load_csv("carnival_si_2215041")
    ca_si = getAvg([ca_si_1, ca_si_2, ca_si_3])
    plt.plot(ca_nips[:, 0], np.convolve(ca_nips[:, col], kernel, mode='same'), label="CA-nips", color=colors[0], linestyle=lines[0])
    plt.plot(ca_da[:, 0], np.convolve(ca_da[:, col], kernel, mode='same'), label="DA", color=colors[1], linestyle=lines[1])
    plt.plot(ca_si[:, 0], np.convolve(ca_si[:, col], kernel, mode='same'), label="SI", color=colors[2], linestyle=lines[2])
    plt.ylabel(ylab[col-1])
    plt.xlabel("Training Epochs")
    plt.title("Playing Carnival")
    plt.legend(loc="best")
    save_plot(savename[col-1])
    plt.show()
    pass


#plot everything
#epoch,num_episodes,total_reward,reward_per_episode,mean_q
def all_the_plots_dem_at(col):
    ylab = ["Number of Episodes", "Total Reward per Epoch", "Average Reward per Episode", "Average Action Value (Q)"]
    savename = ["num_episode_dem_at", "tot_rew_epoch_dem_at", "avg_rew_episode_dem_at", "avg_q_dem_at"]

    da_nips = load_csv("demon_attack-nips")
    da_ca_1 = load_csv("demon_attack_ca_2186052")
    da_ca_2 = load_csv("demon_attack_ca_2215081")
    da_ca_3 = load_csv("demon_attack_ca_2215086")
    da_ca = getAvg([da_ca_1, da_ca_2, da_ca_3])
    da_si_1 = load_csv("demon_attack_si_2186057")
    da_si_2 = load_csv("demon_attack_si_2215095")
    da_si_3 = load_csv("demon_attack_si_2215104")
    da_si = getAvg([da_si_1, da_si_2, da_si_3])
    plt.plot(da_nips[:, 0], np.convolve(da_nips[:, col], kernel, mode='same'), label="DA-nips", color=colors[0], linestyle=lines[0])
    plt.plot(da_ca[:, 0], np.convolve(da_ca[:, col], kernel, mode='same'), label="CA", color=colors[1], linestyle=lines[1])
    plt.plot(da_si[:, 0], np.convolve(da_si[:, col], kernel, mode='same'), label="SI", color=colors[2], linestyle=lines[2])
    plt.ylabel(ylab[col-1])
    plt.xlabel("Training Epochs")
    plt.title("Playing Demon Attack")
    plt.legend(loc="best")
    save_plot(savename[col-1])
    plt.show()
    pass


#plot everything
#epoch,num_episodes,total_reward,reward_per_episode,mean_q
def all_the_plots_sp_in(col):
    ylab = ["Number of Episodes", "Total Reward per Epoch", "Average Reward per Episode", "Average Action Value (Q)"]
    savename = ["num_episode_sp_in", "tot_rew_epoch_sp_in", "avg_rew_episode_sp_in", "avg_q_sp_in"]

    si_nips = load_csv("space_invaders-nips")
    si_ca_1 = load_csv("space_invaders_ca_2186042")
    si_ca_2 = load_csv("space_invaders_ca_2215055")
    si_ca_3 = load_csv("space_invaders_ca_2215056")
    si_ca = getAvg([si_ca_1, si_ca_2, si_ca_3])
    si_da_1 = load_csv("space_invaders_da_2186047")
    si_da_2 = load_csv("space_invaders_da_2215058")
    si_da_3 = load_csv("space_invaders_da_2215059")
    si_da = getAvg([si_da_1, si_da_2, si_da_3])
    plt.plot(si_nips[:, 0], np.convolve(si_nips[:, col], kernel, mode='same'), label="SI-nips", color=colors[0], linestyle=lines[0])
    plt.plot(si_ca[:, 0], np.convolve(si_ca[:, col], kernel, mode='same'), label="CA", color=colors[1], linestyle=lines[1])
    plt.plot(si_da[:, 0], np.convolve(si_da[:, col], kernel, mode='same'), label="DA", color=colors[2], linestyle=lines[2])
    plt.ylabel(ylab[col-1])
    plt.xlabel("Training Epochs")
    plt.title("Playing Space Invaders")
    plt.legend(loc="best")
    save_plot(savename[col-1])
    plt.show()
    pass


if __name__ == "__main__":
    # path = "results/carnival-nips.csv"
    # original(path)
    # all_the_plots_carnival(4)
    # all_the_plots_dem_at(1)
    # all_the_plots_sp_in(1)
    # all_the_plots_sp_in(2)
    # all_the_plots_sp_in(3)
    # all_the_plots_sp_in(4)
    pass

