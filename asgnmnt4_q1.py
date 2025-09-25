# %% [markdown]
# # Assignment 4
# ### Do three of six.

# %% [markdown]
# ### Exercise 1: Contingent Comparisons
# - Load the Minnesota use of force data.
# - Bootstrap the proportion of missing values for `subject_injury` for each race, and plot the results with grouped KDE and ECDF plots
# - Describe what you see. When we consider second order uncertainty, how similar or different are the sampling distributions of these proportions? 

# %%
import pandas as pd
df = pd.read_csv('c:/Users/speed/DS-5030/data/mn_police_use_of_force.csv')
df.head()

# %%
# Bootstrap the proportion of missing values for `subject_injury` for each race
def bs_na_prop(df: pd.DataFrame, var: str):
    props = df.isna().value_counts(normalize=True)
    return props[True]


# %%
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def simulate(df, var, fcn, axes, S=1000, plot=True):
    x = df[var]
    race = df['race'].unique()[0]

    estimates = [fcn(x.sample(frac=1.0,replace=True), var) for s in range (S)]

    if plot:
        sns.kdeplot(estimates, ax = axes[0], label=race).set(title='KDE of Computed Statistics')
        sns.ecdfplot(estimates, ax = axes[1], label=race).set(title='ECDF of Computed Statistics')

    return estimates

# %%
races =  df['race'].dropna().unique()
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
for race in races:
    simulate(df[df['race'] == race], 'subject_injury', bs_na_prop, axes)
axes[0].legend()
axes[1].legend()
plt.show()

# %%
df['race'].value_counts()

# %% [markdown]
# In the KDE, we notice that Other/Mixed race has the lowest rate of missing observations for 'subject_injury', with little overlap from the sampling distribution with the others on the plot. Asian is next lowest, with slight overlap with Native American. White and black have significant overlap and both have distinctly higher mean missing observations and densities. From the KDE, since we observe that the plot of Other/Mixed does not have overlap, that grouping likely has the lowest proportion of missing values compared to the other race groupings. On the other hand, we cannot conclusively determine which has the higher proportion between White and Black due to the significant overlap in the KDE plots. Additionally, Pacific Islander was dropped from the plot because of the 6 observations, all were missing the 'subject_injury" value. While this is a 1.00 proportion of missing values, with such little sample size, no conclusions should be made.
# 
# These conclusions are backed up by the ECDF. As there are significantly more observations of Black and White people in the data, there is less variance in the sample distributions, which is why the KDEs for those race groupings are tighter with higher peaks and their ECDFs are steeper.

# %% [markdown]
# ### Exercise 2: Invitation to Inference
# - Run the simulation code line by line and comment what each line is doing, or write your own code to do the resampling
# - Open the NHANES or Ames prices or College Completion data
# - Use the above function `simulate` to get a sample of estimates for your statistic and your data
# - Create a new function, `interval(L,H,estimates)`, that computes the $L$-th and $H$-th quantiles for your estimates, $H>L$
# - If $L=.05$ and $H=.95$, this is a **90-percent confidence interval**: "For our statistic, this interval captures the true value of the population parameter 90 percent of the time. (We are 90% **confident** that it includes the true value of the parameter, but the probability that the true parameter lies in this interval is 0 or 1.)"
# - We will spend much more time on this later in class, but for people who have done hypothesis testing before, you now know how to do it directly from the data: No central limit theorem required.

# %% [markdown]
# ### Exercise 3: Intro to A/B Testing
# - Go here, and read about this study: https://www.clinicaltrials.gov/study/NCT01985360
# - Read the Study Overview and explain what the goal of the trial is 
# - Read the Study Plan and explain how it was designed and why -- there's lots of medical jargon, but the main point is how patients were assigned to interventions. 
# - Read the Results Posted: Go to **Outcome Measures**. Explain how table 1 ("Incidence of Death from Any Cause or Myocardial Infarction") is a contingency table. These are the data for this exercise.
# - What is the difference in surival rates between the invasive strategy and the conservative strategy?
# - Bootstrap the survival rates for the two groups, and plot them as KDEs and ECDFs against one another
# - Bootstrap the difference in surival rates, and plot it as a KDE and ECDF
# - Is this an effective health intervention? Explain your answer clearly
# 
# This would be what CS people call **A/B testing** and everyone else called a **randomized controlled trial**: Using randomized assignment to detect the difference in outcomes between two groups. (We've just done a non-parametric version of a two-sample t-test.)

# %% [markdown]
# ### Exercise 4: Prediction Uncertainty
# - Pick a dataset and two continuous variables.
# - Recall the LCLS estimator:
# $$
# \hat{y}(z) =  \frac{ \frac{1}{N} \sum_{i=1}^N y_i \times \frac{1}{h}k\left( \frac{z - x_i}{h} \right)}{ \frac{1}{N} \sum_{i=1}^N \frac{1}{h} k\left( \frac{z - x_i}{h} \right)}
# $$
# with the Epanechnikov kernel and the standard plug-in bandwidth for $h$
# - Compute and plot this line for 30 bootstrap samples. Notice where there is a lot of variation in the predictions, versus little variation in the predictions.
# - Now, for any $z$, we can bootstrap a distribution of predictions using the above formula. Do this at the 25th percentile, median, and 75th percentile of $X$.
# - Now, pick a grid for $z$: Obvious choices are all of the unique values in the data, or an equally spaced grid from the minimum value to the maximum value. For each $z$, bootstrap a sample of predictions and compute the .05 and .95 quantiles. Plot these error curves along with your LCLS estimate. Where are your predictions "tight"/reliable? Where are they highly variable/unreliable?

# %% [markdown]
# ### Exercise 5
# - Extend the `kde` class by adding a method to do bandwidth selection using a simple train-test split
# - Extend the `kde` class by adding a method to do bandwidth seleciton by $k$-fold cross validation

# %% [markdown]
# ### Exercise 6
# In this exercise, you're going to do for LCLS what we just did for KDE: Pick the bandwidth 
# 
# Recall, the LCLS estimator is
# $$
# \hat{y}_{N,h}(z) = \dfrac{\frac{1}{N} \sum_{i=1}^N y_i \times \frac{1}{h}k\left(\frac{z-x_i}{h}\right)}{\frac{1}{N} \sum_{i=1}^N \frac{1}{h}k\left(\frac{z-x_i}{h} \right)}.
# $$
# - Select two numeric variables from a data set. Split the data into training and test sets.
# - Write a `predict(X_{train},Y_{train},X_{train})` function/method that takes a set of values $X_{test}$ and data $(X_{train},Y_{train})$, and computed predicted values $\hat{y}(X_{test})$ 
# - Write a function/method that selects the bandwidth by **minimizing** the **mean squared error** of the squared differences between $\hat{y}(x_j)$ and $y_j$ on the test set:
# $$
# MSE(h) = \frac{1}{N_{test}} \sum_{j=1}^{N_{test}} (y_j - \hat{y}_{N_{train},h}(x_j))^2
# $$
# Be sure you understand what's happening here: The training data are used to make predictions for each test observation $x_j$, and then the prediction $\hat{y}(x_j)$ and true value $y_j$ are compared using squared difference

# %% [markdown]
# 


