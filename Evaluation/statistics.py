from scipy.stats import wilcoxon, ttest_ind, mannwhitneyu
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
import csv

# https://www.socscistatistics.com/tests/signedranks/default2.aspx
csv_prefix = "3_"  # change this to "1_" to evaluate files for 1 machine
folder = "overtime16_3m/"   # change this to the desired folder

results_default_0 = pd.read_csv(folder + csv_prefix + "default_action_0.csv")
results_dqn = pd.read_csv(folder + csv_prefix + "dqn_eval.csv")
results_ppo = pd.read_csv(folder + csv_prefix + "ppo_eval.csv")
results_rnd = pd.read_csv(folder + csv_prefix + "rnd_action.csv")


base_sample = results_default_0.total_cost
sample_to_test = results_ppo.total_cost


# Value to test p values against (Signifikanzniveau)
alpha = 0.05
# Set up dataframe
df = pd.DataFrame(list(zip(base_sample,sample_to_test)), columns =['Default','Testsample'])

# Run Shapiro-Wilk to test for normality
stat_shapiro_base, p_shapiro_base = stats.shapiro(base_sample)
stat_shapiro_test, p_shapiro_test = stats.shapiro(sample_to_test)

if p_shapiro_base > alpha:
    print("shapiro on base sample: fail to reject H0, sample has normal distribution")
else:
    print("shapiro on base sample: reject H0")
if p_shapiro_test > alpha:
    print("shapiro on test sample: fail to reject H0, sample has normal distribution")
else:
    print("shapiro on test sample: reject H0")


# Run wilcoxon signed rank test ->for paired samples
stat_wil, p_value_wil = wilcoxon(base_sample, sample_to_test)
# Run t test
stat_ttest, p_value_ttest = ttest_ind(base_sample, sample_to_test)

if p_value_wil > alpha:
    print("wilcoxon: Same distribution, fail to reject H0, no significant difference, "
          "any difference between the two samples is due to chance ")
else:
    print("wilcoxon: Different distribution, reject H0, significant difference")


if p_value_ttest > alpha:
    print("ttest: Same distribution, fail to reject H0, no significant difference")
else:
    print("ttest: Different distribution, reject H0, significant difference")



# data_to_plot = [sample_default0, sample_aradqn, sampe_rnd]
#

# Create QQ plot
# arr1 = np.array(sample_default0)
# arr2 = np.array(sample_aradqn)
# pp_x = sm.ProbPlot(arr1)
# pp_y = sm.ProbPlot(arr2)
# qqplot_2samples(pp_x, pp_y)
# plt.show()


# # Create a figure instance
# fig = plt.figure(figsize =(10, 7))
# #Create the boxplot
# plt.boxplot(data_to_plot)
# #show plot
# plt.show()
# #Save the figure
# fig.savefig('boxplot.png', bbox_inches='tight')



# stats.probplot(sample_default0, dist="norm", plot=plt)
# plt.title("Default vs ara dqn")
# plt.savefig("jobshop.png")

# if __name__ == "__main__":
#     answer = input('Type...to... \n'
#                    'Hypothesis tests:\n'
#                    '"a" ... run Mann-Whitney U test \n'
#                    '"b" ... run t-test \n'
#                    '"c" ... run Shapiro-Wilk test \n'
#                    'Plots:\n'
#                     'QQ plot\n'
#                     'histogram\n'
#                    )
#     if answer == "a" or answer == "":
#         print("test")

def plot_scatter_training(base_folder):
    # folder = "overtime64_3m/"
    folder = base_folder + "Training/"

    rewards_per_period = pd.read_csv(folder + "rewards_per_period.csv",sep='\\t', engine='python')
    #q_values_for_fixed_obs = pd.read_csv(folder + "q_values_learned_results.csv")

    # Set up dataframe
    period = rewards_per_period.Period
    reward = rewards_per_period.Reward
    rho = rewards_per_period.Rho
    df = pd.DataFrame(list(zip(period,reward)), columns =['Period','Reward'])

    #plt.scatter(period, reward,)
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(period, reward)
    ax.set_xlabel('Period')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards per period')
    plt.show()
    return
#plot_scatter_training(folder)
