# Co2 and CH4 Emissions from around the world

This is a project to visualize the co2 emissions from around the world. The data is from the
[EDGAR](https://edgar.jrc.ec.europa.eu/dataset_ghg70). The data is from 2000-2021.

# Data Questions

1. Is there a correlation between the co2 emissions and CH4 emissions?
    1. How strong is the correlation?
2. Is there a correlation between the co2 emissions and the population for specific countries?
3. Can we predict the co2 emissions for the next 10 years?

# Implemented approaches

1. Examination of the correlation
    + A correlation was found across all test countries, in some cases with a high R^2 value of 0.569 for the United
      Kingdom. In contrast, countries that strive to reduce emissions have a lower value. It can be assumed that CH4 is
      reduced before CO2 and thus the relationship is broken. To prove the correlation, the method of person correlation
      was used. The Pearson correlation coefficient measures the linear
      relationship between two datasets. Like other correlation coefficients, this one varies between -1 and +1 with 0
      implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. Positive correlations imply
      that as x increases, so does y. Negative correlations imply that as x increases, y decreases

2. Examination of the correlation between the population and the co2 emissions
    + The same methodology was used as in question 1.
      For all test countries, it can be seen that the result is a negative number. In some cases, values of -0.891 (
      Belgium) are achieved. The reason for this is the growing CO2 emissions with a dwindling population.

3. Prediction of the co2 emissions for the next 1 year
    + Three approaches were taken to make a prediction.
      One is the Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model with two
      configurations and the other is the Autoregressive Integrated Moving Average method.
      The SARIMAX provided the most accurate results.
        The SARIMAX model is an extension of the ARIMA model that explicitly supports univariate time series data with
        exogenous variables.
