from scipy.stats import wilcoxon, ttest_rel
import pandas as pd
import csv
from scipy import stats
import matplotlib.pyplot as plt


# https://www.socscistatistics.com/tests/signedranks/default2.aspx
machine_prefix = "_3"  # change this to "_1" to evaluate files for 1 machine
file_suffix = machine_prefix + "_env_metrics_per_episode.csv"
folder = "bil2-A/"  # change this to the desired folder (subfolder of /Evaluation/), e.g. "bil3-B/"


def get_metrics_per_episode():
    results_default_0 = pd.read_csv(folder + "default0" + file_suffix)
    results_default_1 = pd.read_csv(folder + "default1" + file_suffix)
    results_default_2 = pd.read_csv(folder + "default2" + file_suffix)
    results_dqn = pd.read_csv(folder + "ARA_DIRL_eval" + file_suffix)
    results_ppo = pd.read_csv(folder + "ppo_eval" + file_suffix)
    results_rnd = pd.read_csv(folder + "random" + file_suffix)
    results_custom = pd.read_csv(folder + "custom" + file_suffix)
    return results_default_0, results_default_1, results_default_2, results_dqn, results_ppo, results_rnd, results_custom


def get_statistical_test_results():
    """
    Run multiple statistical tests to check for significant differences between the samples.
    Change the sample variables according to the agents that you want to evaluate.
    Each sample variable has two parts, one is the sample name (results_default_0) and the other
    is the column which should be analyzed (total_cost)
    """
    results_default_0, results_default_1, results_default_2, results_dqn, \
        results_ppo, results_rnd, results_custom, results_custom = get_metrics_per_episode()

    # Potential values to inspect:
    # total_cost, wip_cost, fgi_cost, lateness_cost, amount_of_shipped_orders, bottleneck_utilization, \
    # late_orders, early_orders, sum_of_lateness, sum_of_tardiness, average_flow_time
    base_sample = results_ppo.total_cost
    sample_to_test = results_custom.total_cost

    # Value to test p values against (Signifikanzniveau)
    alpha = 0.05
    # Set up dataframe
    df = pd.DataFrame(list(zip(base_sample, sample_to_test)), columns=['Default', 'Testsample'])

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
    stat_ttest, p_value_ttest = ttest_rel(base_sample, sample_to_test)

    if p_value_wil > alpha:
        print("wilcoxon: Same distribution, fail to reject H0, no significant difference, "
              "any difference between the two samples is due to chance ")
    else:
        print("wilcoxon signed rank: Different distribution, reject H0, significant difference")

    if p_value_ttest > alpha:
        print("paired t-test: Same distribution, fail to reject H0, no significant difference")
    else:
        print("paired t-test: Different distribution, reject H0, significant difference")
    return


def get_average_metrics_per_episode():
    """
    Change the sample variable according to the agent that you want to evaluate
    """
    with open(str(folder) + "metrics.csv", mode='w') as metrics_CSV:
        results_writer = csv.writer(metrics_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['Sample', 'total_cost', 'wip_cost', 'fgi_cost',
                                         'lateness_cost', 'overtime_cost', 'amount_of_shipped_orders',
                                         'bottleneck_utilization', 'late_orders', 'early_orders',
                                         'sum_of_lateness', 'sum_of_tardiness', 'average_flow_time', 'service_level'])
        metrics_CSV.close()

    results_default_0, results_default_1, results_default_2, results_dqn, results_ppo, results_rnd, results_custom \
        = get_metrics_per_episode()
    list_of_samples = [results_default_0, results_default_1, results_default_2, results_rnd, results_dqn,
                       results_ppo, results_custom]
    sample_names = ["d0","d1","d2","rnd","dqn","ppo","custom"]

    for index, sample in enumerate(list_of_samples):
        # Service level = percentage of orders delivered on time
        service_level = 1 - (sample.late_orders / sample.amount_of_shipped_orders)
        # Write values to CSV
        with open(str(folder) + "metrics.csv", mode='a') as metrics_CSV:
            results_writer = csv.writer(metrics_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow([sample_names[index],
                                     round(sample.mean(), 0)[1],
                                     round(sample.mean(), 0)[2],
                                     round(sample.mean(), 0)[3],
                                     round(sample.mean(), 0)[4],
                                     round(sample.mean(), 0)[5],
                                     round(sample.mean(), 0)[6],
                                     round(sample.mean(), 2)[7],
                                     round(sample.mean(), 0)[8],
                                     round(sample.mean(), 0)[9],
                                     round(sample.mean(), 0)[10],
                                     round(sample.mean(), 0)[11],
                                     round(sample.mean(), 0)[12],
                                     round(service_level.mean(), 2)])
            metrics_CSV.close()
    return


def plot_rewards_per_period():
    import statsmodels.api as sm
    from statsmodels.graphics.gofplots import qqplot_2samples
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

    return


def plot_scatter_training(base_folder):
    # folder = "overtime64_3m/"
    folder = base_folder + "Training/"

    rewards_per_period = pd.read_csv(folder + "rewards_per_period.csv", sep='\\t', engine='python')
    # q_values_for_fixed_obs = pd.read_csv(folder + "q_values_learned_results.csv")

    # Set up dataframe
    period = rewards_per_period.Period
    reward = rewards_per_period.Reward
    rho = rewards_per_period.Rho
    df = pd.DataFrame(list(zip(period, reward)), columns=['Period', 'Reward'])

    # plt.scatter(period, reward,)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(period, reward)
    ax.set_xlabel('Period')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards per period')
    plt.show()
    return


if __name__ == "__main__":
    answer = input('Type...to... \n'
                   '"a" ... run statistical tests \n'
                   '"b" ... create scatter plot of rewards per period (not working yet) \n'
                   '"c" ... Write average metrics per episode to CSV\n'
                   'Note that there need to be csv files for all agents, \n'
                   'else there will be an error. To fix it remove the unused agents. \n'
                   )
    if answer == "a":
        get_statistical_test_results()
    if answer == "b":
        plot_scatter_training(folder)
    if answer == "c":
        get_average_metrics_per_episode()
