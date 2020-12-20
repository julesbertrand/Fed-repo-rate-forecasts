# Forecasting the Effective Fed Interest rate using Statistical Learning and Macro-Economic Features.

![GitHub](https://img.shields.io/github/license/julesbertrand/Fed-repo-rate-forecasts)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/) 
<!-- [![Travis (.com)](https://img.shields.io/travis/com/julesbertrand/litreading-insight-project?label=TravisCI)](https://travis-ci.com/github/julesbertrand/litreading-insight-project) -->

## Background and Goal of this Project

This project was started as a follow-up to an academic project for Statistical Learning class. 

In a world where transactions take place at the speed of light, banks constantly need to balance their cash reserves. Banks lend or borrow reserve balances from other banking institutions every day, without any collateral other than that of the central bank, in order to be exactly at the mandatory reserve rate set by law. The effective federal funds rate is the weighted average of all interest rates negotiated between the borrowing and lending banks. The target is set by the Federal Reserve, which intervenes in the markets to ensure that the target is met. In Europe, for example, the ECB has been intervening constantly for several years via Quantitative easing.

This project focuses on the prediction of the effective interest rate managed by the U.S. Federal Reserve, based on past macroeconomic and financial indicators.

## Table of Content
* [Background](#background-and-goal-of-this-project)
* [Fed Repo rate: What is it?](#fed-repo-rate:-what-is-it?)
* [Data](#data)
* [Feature Engineering](#feature-engineering)
* [Metrics and Models](#metrics-and-models)
* [Forecasts](#forecasts)
* [Wrapping-up, Insights and Next Steps](#wrapping-up,-insights-and-next-steps)


## Fed Repo rate: What is it?

Everyday, banks are creating money by lending money to people. All these loans are granted with several conditions, including the interest rate, which represents the price at which the bank is willing to lend money to you. The higher the interest rate, the less money you will borrow. 

But how is this individual interest rate determined ? The banks are mainly interested in three criteria, which I list here in no particular order. Firstly, they take into account the borrower's situation: a higher risk of credit default implies a higher cost of borrowing and therefore a higher interest rate. Secondly, banks are dealing with the macroeconomic situation. If the economy is in crisis the number of defaults is higher than usual, so the cost of credit rises. Last but not least, banks have to consider their own situation. Depending on its resources, and the criteria laid down by the authorities, they will be able to lend more, which will decrease the cost of credit, or less, which will led to an increased cost of credit. Among these criteria are reserve requirements, capital requirements, and… the main refinancing rate.

When a bank is lending money, it needs to adjust the level of its reserve account to meet central bank reserve ratio requirement, currently set at 10% of deposits. This reserve cash is to be used by the financial institution to meet any unexpected or large demand of withdrawals or any huge credit default episode such as in 2008. To put it simple if a bank is lending $1M, it must increase its reserve by 10% * $1M = $100k. In order to meet central bank requirements, the banks with excess reserve cash at the end of the day can lend reserve cash to other banks at a rate fixed by the Central Bank: the main refinancing rate, which is known in the US as the Federal funds rate.

From now on I will mainly use interest rate, repo rate, Fed rate to refer to the American Federal funds rate.
The Fed rate is therefore the rate at which banks borrow from each other in order to always meet some of the Fed's criteria. The lower it is, the more banks will want to borrow from each other, and the easier it will be for them to lend to their clients, and vice versa. Thus this rate is, among other things, an instrument for controlling the money supply, the amount of debt, and inflation in the economy. It has a direct impact on the interest rate paid by the average consumer who wants to take out a home loan, for example, but also on loans granted to companies. Being able to predict it means being able to anticipate some movements in interest rates levels.

Now that we know what this rate is and what impact it can have, the next questions are why and how does the Federal Bank decide on its value ? As stated on its website:

 > *The Congress has directed the Fed to conduct the nation's monetary policy to support three specific goals: maximum sustainable employment, stable prices, and moderate long-term interest rates. These goals are sometimes referred to as the Fed's "mandate".*

It mainly means that the Fed must use the tools at its disposal in order to ensure low unemployment rate and steady, controlled inflation. However, deciding on the interest rate is not an easy job as the concerns are diverse: employment, prices and wages, consumer spending, business investments, Foreign exchange indicators, global health of economy, ... But we can still distinguish two main patterns that are repeated in the decisions of all major central banks (Fed, ECB, Bank of Japan, Bank of England, …). When an economic crisis is surging or when the inflation is low, central banks usually lower the rates to boost borrowing, and therefore investment and/or price rises. When the economy overheats, i.e. inflation is too high or unemployment is structurally very low, it is the other way around: rates go up. This is easy to see when you look at the interest rates curve below.

![Fed effective rate graph](resources/fredgraph.png?raw=true)
Federal funds rate since 1955 (1). Shaded areas correspond to periods of recessions as defined by the [US National Bureau of Economic Research (NBER)]("https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions"). The NBER announced on June 8, 2020 that a recession began in February 2020, time around which we can see the Fed funds rate going down even before the Covid-19 pandemic hit.

## Data

As mentioned above, the Fed uses macroeconomic and financial indicators to set the repo rate eight times a year. I chose to build as exhaustive a list of indicators as possible, and then reduce the number of characteristics used to build the models. I first looked at what the Fed officially takes into account and I then added some indicators that I thought were relevant. The list of all data series, where they come from, and their dates are listed in the table below.

![Data index table](resources/data_index.png?raw=true)

## Feature engineering
### 1. Stationarity of time-series: Dickey-Fuller test
### 2. Seasonality
### 3. What data to rely on
### 4. Choice of start and end date
### 5. Feature correlation

## Metrics and Models

Metrics: RMSE

Prediction of change in fed interest rate, then add to current interest rate to get interest rate prediction

Models: ensemble, regression

Next: try neural net

## Forecasts

![forecasts](resources/forecasts.png?raw=true)

## Wrapping-up, Insights and Next Steps



-----

[1] Board of Governors of the Federal Reserve System (US), Effective Federal Funds Rate [FEDFUNDS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/FEDFUNDS, December 19, 2020.

[2] https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions

[3] https://www.lafinancepourtous.com/decryptages/finance-perso/banque-et-credit/taux-d-interet/comment-se-fabrique-un-taux-dinteret/

[4] https://www.investopedia.com/terms/r/reserveratio.asp

[5] https://www.federalreserve.gov/faqs/what-economic-goals-does-federal-reserve-seek-to-achieve-through-monetary-policy.htm

