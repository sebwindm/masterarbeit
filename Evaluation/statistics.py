from scipy.stats import wilcoxon, ttest_ind, mannwhitneyu
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples

# https://www.socscistatistics.com/tests/signedranks/default2.aspx

sample_default0 = [2251313,
                   2285646,
                   2240579,
                   2329507,
                   2175986,
                   2313574,
                   2320546,
                   2183954,
                   2323703,
                   2280200,
                   2268690,
                   2287089,
                   2267923,
                   2306972,
                   2304844,
                   2262520,
                   2268984,
                   2326529,
                   2313792,
                   2248504,
                   2304034,
                   2260305,
                   2320244,
                   2373406,
                   2281547,
                   2300410,
                   2260874,
                   2281053,
                   2294764,
                   2256631
                   ]
sample_aradqn = [1969808.0,
                 1981983.0,
                 1970840.0,
                 1922712.0,
                 1972973.0,
                 1982638.0,
                 1971125.0,
                 1910179.0,
                 1971219.0,
                 1975960.0,
                 1921461.0,
                 1924158.0,
                 1940930.0,
                 1940070.0,
                 1937120.0,
                 1952867.0,
                 1933464.0,
                 1998999.0,
                 1945520.0,
                 1900738.0,
                 1950475.0,
                 1946792.0,
                 1979347.0,
                 1956603.0,
                 1957269.0,
                 1960082.0,
                 1947880.0,
                 1930803.0,
                 1928397.0,
                 1982507.0
                 ]
sampe_rnd = [1937436.0,
1932774.0,
1920218.0,
1965444.0,
1897143.0,
1986429.0,
1907595.0,
1891451.0,
1961440.0,
1920215.0,
1959918.0,
1968296.0,
1953729.0,
1950984.0,
1903104.0,
1936186.0,
1898337.0,
1903204.0,
1908082.0,
1981604.0,
1936926.0,
1933223.0,
1968439.0,
1952512.0,
1936433.0,
1914845.0,
1942910.0,
1911482.0,
1946532.0,
1914898.0
]

base_sample = sample_default0
sample_to_test = sample_aradqn

# Value to test p values against (Signifikanzniveau)
alpha = 0.05
# Set up dataframe
df = pd.DataFrame(list(zip(base_sample,sample_to_test)), columns =['Default','Testsample'])
#print(df.describe())

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


# Run mann whitney u test ->for independent samples
stat_mwu, p_value_mwu = mannwhitneyu(base_sample, sample_to_test)
# Run wilcoxon signed rank test ->for paired samples
stat_wil, p_value_wil = wilcoxon(base_sample, sample_to_test)
# Run t test
stat_ttest, p_value_ttest = ttest_ind(base_sample, sample_to_test)

if p_value_mwu > alpha:
    print("mannwhitneyu: Same distribution, fail to reject H0, no significant difference, "
          "any difference between the two samples is due to chance ")
else:
    print("mannwhitneyu: Different distribution, reject H0, significant difference")


if p_value_ttest > alpha:
    print("ttest: Same distribution, fail to reject H0, no significant difference")
else:
    print("ttest: Different distribution, reject H0, significant difference")



data_to_plot = [sample_default0, sample_aradqn, sampe_rnd]


# Create QQ plot
# arr1 = np.array(sample_default0)
# arr2 = np.array(sample_aradqn)
# pp_x = sm.ProbPlot(arr1)
# pp_y = sm.ProbPlot(arr2)
# qqplot_2samples(pp_x, pp_y)
# plt.show()


# Create a figure instance
fig = plt.figure(figsize =(10, 7))
#Create the boxplot
plt.boxplot(data_to_plot)
#show plot
plt.show()
#Save the figure
fig.savefig('boxplot.png', bbox_inches='tight')



# stats.probplot(sample_default0, dist="norm", plot=plt)
# plt.title("Default vs ara dqn")
# plt.savefig("jobshop.png")