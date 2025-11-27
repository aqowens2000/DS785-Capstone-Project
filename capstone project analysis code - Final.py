#Import libraries
from bs4 import BeautifulSoup as bs
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm
import textwrap
from scipy import stats

### Data Loading and Processing
# This section performs the remaining data extract, transform, and load processes for the project
#Loading in the data and webscraping sites
requester = {'User-Agent': "Andrew Owens - Des Moines, IA"}
stock_df = pd.read_csv("Current Stock Data.csv")
historical_stock_prices = pd.read_csv("Historical Stock Prices.csv")
historical_stock_returns = pd.read_csv("Historical Stock Returns.csv")

SP500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MemeStock_url = "https://www.quiverquant.com/scores/memestocks"

#Getting a listing of SP500 tickers
webpage = requests.get(SP500_url, headers = requester)
page = bs(webpage.content, features = 'lxml')
table_data = page.find_all('table', {'class':"wikitable"})
table_list = pd.read_html(str(table_data), encoding = 'utf-8')
SP500_table = table_list[0]
SP500_tickers = SP500_table['Symbol']

#Getting a listing of Meme Stocks from WallStreetBets
webpage = requests.get(MemeStock_url, headers = requester)
page = bs(webpage.content, features = 'lxml')
table_data = page.find_all('table', {'id':"myTable"})
table_list = pd.read_html(str(table_data), encoding = 'utf-8')
MemeStock_table = table_list[0]
MemeStock_tickers_v1 = MemeStock_table['Ticker']
MemeStock_tickers_v2 = MemeStock_tickers_v1.str.split('  ', expand = True)
MemeStock_tickers = MemeStock_tickers_v2[0]

#Create new fields whether the ticker is in the S&P500 or a meme stock
stock_df['SP500'] = stock_df['Ticker'].isin(SP500_tickers).astype(int)
stock_df['Meme'] = stock_df['Ticker'].isin(MemeStock_tickers).astype(int)

#Filter out tickers with every field other than name as blank - excluded from analysis
stock_df_v2 = stock_df[(stock_df['P/E'].notnull()) | (stock_df['Industry'].notnull()) | 
                       (stock_df['Sector'].notnull()) | (stock_df['Beta'].notnull()) | 
                       (stock_df['Market Cap'].notnull()) | (stock_df['Price/Book'].notnull()) | 
                       (stock_df['DividendYield'].notnull())]

### Converting the Stock Performance Data from $ prices to percent returns

#Compounding Percent Change Formula
def ln_pct_change(prev, new):
    if math.isnan(prev) or math.isnan(new):
        percent_change = np.nan
    elif prev <= 0 or new <= 0:
        percent_change = np.nan
    else:
        percent_change = math.log(new/prev) * 100
    return percent_change

year = 1998 #Adjust for Ticker column and 1999 starting column
historical_prices = pd.DataFrame()
historical_returns = pd.DataFrame()

# Renaming some columns and data types for cleaner data frames
historical_stock_prices.rename(columns = {'Unnamed: 0': 'Ticker'}, inplace = True)
historical_stock_prices['Ticker'] = historical_stock_prices['Ticker'].astype('string')
historical_stock_returns.rename(columns = {'Unnamed: 0': 'Ticker'}, inplace = True)
historical_stock_returns['Ticker'] = historical_stock_returns['Ticker'].astype('string')

historical_stock_prices[historical_stock_prices.columns[1:]] = historical_stock_prices[historical_stock_prices.columns[1:]].clip(lower = 0.0001) # This is to prevent negatives from showing up
historical_stock_returns[historical_stock_returns.columns[1:]] = historical_stock_returns[historical_stock_returns.columns[1:]].clip(lower = 0.0001) # This is to prevent negatives from showing up

#Loop through each column in the historical prices/returns dataframes and create new dataframes based on percentages
for year_col in historical_stock_prices:
    if year_col == 'Ticker':
        historical_prices['Ticker'] = historical_stock_prices[year_col]
        historical_returns['Ticker'] = historical_stock_returns[year_col]
        
    elif year_col == '12/31/1999':
        prev_col = year_col
        
    else:
        historical_prices[str(year)] = historical_stock_prices.apply(lambda row: ln_pct_change(row[prev_col], row[year_col]), axis = 1)
        historical_returns[str(year)] = historical_stock_returns.apply(lambda row: ln_pct_change(row[prev_col], row[year_col]), axis = 1)
        prev_col = year_col
        
    year += 1

### Historical Performance Processing

Price_mean = historical_prices.iloc[:, 1:].mean(axis = 0)
Price_volatility = historical_prices.iloc[:, 1:].std(axis = 0)

historical_prices.loc[len(historical_prices.index)] = Price_mean
historical_prices.loc[len(historical_prices.index)-1, 'Ticker'] = "Ave Market Return" 
historical_prices.loc[len(historical_prices.index)] = Price_volatility
historical_prices.loc[len(historical_prices.index)-1, 'Ticker'] = "Ave Market Volatility"
historical_prices.set_index('Ticker', inplace = True)

#Plot of Average Market Returns
fig, ax = plt.subplots(1,1)

plt.scatter(historical_prices.loc['Ave Market Return'].index, historical_prices.loc['Ave Market Return'], color = 'blue')
plt.xlabel("Year")
plt.ylabel("Average Return")
plt.axhline(y = 0, color = 'r', linewidth = 2)
ax.tick_params(axis = 'x',  labelrotation = 90)
plt.title("Average Market Return 2000 - 2024")
lower_bound = -200 #Implies a drop of 86%
upper_bound = 200 #Implies a more than 6x increase
historical_prices.clip(lower_bound, upper_bound, inplace = True)

Price_mean = historical_prices.iloc[:, 1:].mean(axis = 0)
Price_volatility = historical_prices.iloc[:, 1:].std(axis = 0)

historical_prices.loc["Ave Market Return"] = Price_mean
historical_prices.loc["Ave Market Volatility"] = Price_volatility

# Plot of Average Mrket Returns after applying trimming
fig, ax = plt.subplots(1,1)

plt.scatter(historical_prices.loc['Ave Market Return'].index, historical_prices.loc['Ave Market Return'], color = 'blue')
plt.xlabel("Year")
plt.ylabel("Average Return")
plt.axhline(y = 0, color = 'r', linewidth = 2)
ax.tick_params(axis = 'x',  labelrotation = 90)
plt.title("Average Market Return 2000 - 2024")

average_price = historical_prices.mean(axis = 1)
volatility_price = historical_prices.std(axis = 1)
sharpe_ratios = (average_price - 4) / volatility_price

# Boxplot illustration distribution of stock returns
fig, ax = plt.subplots(1,1)
sns.boxplot(x = average_price)
plt.title("Box Plot of Average Stock Returns")
plt.show()

historical_prices['2000-2024 Average Returns'] = average_price
historical_prices['2000-2024 Average Volatility'] = volatility_price
historical_prices['2000-2024 Sharpe Ratio'] = sharpe_ratios

historical_prices['2000-2024 Sharpe Ratio'].describe()
historical_prices_v2 = historical_prices[(historical_prices['2000-2024 Sharpe Ratio'].notnull()) & (historical_prices['2000-2024 Average Volatility'] != 0)]
historical_prices_v2['2000-2024 Sharpe Ratio'].describe()
historical_prices_v2[(historical_prices_v2['2000-2024 Sharpe Ratio'] < -5) | (historical_prices_v2['2000-2024 Sharpe Ratio'] > 5)] # These 24 obs. are outliers and will be removed from analysis
historical_prices_v2 = historical_prices_v2[(historical_prices_v2['2000-2024 Sharpe Ratio'] > -5) & (historical_prices_v2['2000-2024 Sharpe Ratio'] < 5)]

# Create the Training 2000-2014 and Test 2015-2024 splits
training_period = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014"]
test_period = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]

average_price_2014 = historical_prices_v2[training_period].mean(axis=1)
volatility_price_2014 = historical_prices_v2[training_period].std(axis=1)
sharpe_ratios_2014 = (average_price_2014 - 4) / volatility_price_2014

average_price_2024 = historical_prices_v2[test_period].mean(axis=1)
volatility_price_2024 = historical_prices_v2[test_period].std(axis=1)
sharpe_ratios_2024 = (average_price_2024 - 4) / volatility_price_2024

historical_prices_v2['2000-2014 Average Returns'] = average_price_2014
historical_prices_v2['2000-2014 Average Volatility'] = volatility_price_2014
historical_prices_v2['2000-2014 Sharpe Ratio'] = sharpe_ratios_2014

historical_prices_v2['2015-2024 Average Returns'] = average_price_2024
historical_prices_v2['2015-2024 Average Volatility'] = volatility_price_2024
historical_prices_v2['2015-2024 Sharpe Ratio'] = sharpe_ratios_2024

historical_prices_v2['2000-2014 Sharpe Ratio'].replace(-np.inf, np.nan, inplace = True)
historical_prices_v2['2015-2024 Sharpe Ratio'].replace(-np.inf, np.nan, inplace = True)

historical_prices_v2['2000-2014 Sharpe Ratio'].clip(-5, 5, inplace = True)
historical_prices_v2['2015-2024 Sharpe Ratio'].clip(-5, 5, inplace = True)

historical_prices_v3 = historical_prices_v2
historical_prices_v3.describe()
historical_prices_v3.dtypes

### Same analysis as above, except for otal returns

Return_mean = historical_returns.iloc[:, 1:].mean(axis = 0)
Return_volatility = historical_returns.iloc[:, 1:].std(axis = 0)

historical_returns.loc[len(historical_returns.index)] = Return_mean
historical_returns.loc[len(historical_returns.index)-1, 'Ticker'] = "Ave Market Return" 
historical_returns.loc[len(historical_returns.index)] = Return_volatility
historical_returns.loc[len(historical_returns.index)-1, 'Ticker'] = "Ave Market Volatility"
historical_returns.set_index('Ticker', inplace = True)

# Plot of Average Market Returns
fig, ax = plt.subplots(1,1)

plt.scatter(historical_returns.loc['Ave Market Return'].index, historical_returns.loc['Ave Market Return'], color = 'blue')
plt.xlabel("Year")
plt.ylabel("Average Return")
plt.axhline(y = 0, color = 'r', linewidth = 2)
ax.tick_params(axis = 'x',  labelrotation = 90)
plt.title("Average Market Return w/ Dividends 2000 - 2024")
lower_bound = -200 #Implies a drop of 86%
upper_bound = 200 #Implies a more than 6x increase
historical_returns.clip(lower_bound, upper_bound, inplace = True)

Return_mean = historical_returns.iloc[:, 1:].mean(axis = 0)
Return_volatility = historical_returns.iloc[:, 1:].std(axis = 0)

historical_returns.loc["Ave Market Return"] = Return_mean
historical_returns.loc["Ave Market Volatility"] = Return_volatility

#Plot of Average Market Returns after trimming
fig, ax = plt.subplots(1,1)

plt.scatter(historical_returns.loc['Ave Market Return'].index, historical_returns.loc['Ave Market Return'], color = 'blue')
plt.xlabel("Year")
plt.ylabel("Average Return")
plt.axhline(y = 0, color = 'r', linewidth = 2)
ax.tick_params(axis = 'x',  labelrotation = 90)
plt.title("Average Market Return w/ Dividends 2000 - 2024")

average_return = historical_returns.mean(axis = 1)
volatility_return = historical_returns.std(axis = 1)
sharpe_ratios_return = (average_return - 4) / volatility_return

fig, ax = plt.subplots(1,1)
sns.boxplot(x = average_return)
plt.title("Box Plot of Average Stock Returns w/ Dividends")
plt.show()

historical_returns['2000-2024 Average Returns'] = average_return
historical_returns['2000-2024 Average Volatility'] = volatility_return
historical_returns['2000-2024 Sharpe Ratio'] = sharpe_ratios_return

historical_returns['2000-2024 Sharpe Ratio'].describe()
historical_returns_v2 = historical_returns[(historical_returns['2000-2024 Sharpe Ratio'].notnull()) & (historical_returns['2000-2024 Average Volatility'] != 0)]
historical_returns_v2['2000-2024 Sharpe Ratio'].describe()
historical_returns_v2[(historical_returns_v2['2000-2024 Sharpe Ratio'] < -5) | (historical_returns_v2['2000-2024 Sharpe Ratio'] > 5)] # These 27 obs. are outliers and will be removed from analysis
historical_returns_v2 = historical_returns_v2[(historical_returns_v2['2000-2024 Sharpe Ratio'] > -5) & (historical_returns_v2['2000-2024 Sharpe Ratio'] < 5)]

average_return_2014 = historical_returns_v2[training_period].mean(axis=1)
volatility_return_2014 = historical_returns_v2[training_period].std(axis=1)
sharpe_ratios_2014_return = (average_return_2014 - 4) / volatility_return_2014

average_return_2024 = historical_returns_v2[test_period].mean(axis=1)
volatility_return_2024 = historical_returns_v2[test_period].std(axis=1)
sharpe_ratios_2024_return = (average_return_2024 - 4) / volatility_return_2024

historical_returns_v2['2000-2014 Average Returns'] = average_return_2014
historical_returns_v2['2000-2014 Average Volatility'] = volatility_return_2014
historical_returns_v2['2000-2014 Sharpe Ratio'] = sharpe_ratios_2014_return

historical_returns_v2['2015-2024 Average Returns'] = average_return_2024
historical_returns_v2['2015-2024 Average Volatility'] = volatility_return_2024
historical_returns_v2['2015-2024 Sharpe Ratio'] = sharpe_ratios_2024_return

historical_returns_v2['2000-2014 Sharpe Ratio'].replace(-np.inf, np.nan, inplace = True)
historical_returns_v2['2015-2024 Sharpe Ratio'].replace(-np.inf, np.nan, inplace = True)

historical_returns_v2['2000-2014 Sharpe Ratio'].clip(-5, 5, inplace = True)
historical_returns_v2['2015-2024 Sharpe Ratio'].clip(-5, 5, inplace = True)

#Make sure both dataframes have the same stock records
historical_returns_v2 = historical_returns_v2[historical_returns_v2.index.isin(historical_prices_v3.index)] # For consistency
historical_returns_v3 = historical_returns_v2
historical_returns_v3.describe()
historical_returns_v3.dtypes

#Make sure that all dataframes that have only the same tickers and any filtered out tickers on one is filtered out on the rest

stock_df_v2 = stock_df_v2[(stock_df_v2['Ticker'].isin(historical_prices_v3.index)) & (stock_df_v2['Ticker'].isin(historical_returns_v3.index))]
historical_prices_v4 = historical_prices_v3[historical_prices_v3.index.isin(stock_df_v2['Ticker'])]
historical_returns_v4 = historical_returns_v3[historical_returns_v3.index.isin(stock_df_v2['Ticker'])]


### Exploratory Analysis
#This section performs univariate and multivariate analysis of the features available within the datasets
#The charts used in presentations 2 and 3 come from this section of code

#Demographic Univariate Analysis

# P/E Field
print(stock_df_v2['P/E'].describe()) # infinite values and extremely high/low values are nonsensical
stock_df_v2['P/E'] = stock_df_v2['P/E'].replace(np.inf, np.nan)
print(stock_df_v2['P/E'].describe())

sns.boxplot(x = stock_df_v2['P/E'])
plt.title("Box Plot of P/E")
plt.show() # Lots of outliers on both sides

Neg_PE = stock_df_v2[stock_df_v2['P/E'] <= -100]
len(Neg_PE) / len(stock_df_v2) # Less than 1% of the data affected
stock_df_v2['P/E'][stock_df_v2['P/E'] <= -100] = -100 #Set a floor at -100
High_PE = stock_df_v2[stock_df_v2['P/E'] >= 100] # Less than 1.4% of the data affected
len(High_PE) / len(stock_df_v2) # Less than 1.5% of the data affected
stock_df_v2['P/E'][stock_df_v2['P/E'] >= 100] = 100 #Set a ceiling at 100

print(stock_df_v2['P/E'].describe())

#Industry
industries = stock_df_v2['Industry'].value_counts()
top_industries = industries.head(15) #only looking at the top 15 for graph since there are a wide number

fig, ax = plt.subplots(1,1)
plt.hist(top_industries.index, bins = len(top_industries), weights = top_industries.values, color = 'lightblue')
plt.title("Top 15 Most Frequent U.S. Industries")
plt.xlabel("Industry")
plt.ylabel("Freq")
plt.xticks(rotation = 45, ha = 'right')
plt.show()

#Sector
sectors = stock_df_v2['Sector'].value_counts() #There are only 11 sectors. Industry is a subset of Sector

fig, ax = plt.subplots(1,1)
plt.hist(sectors.index, bins = len(sectors), weights = sectors.values, color = 'darkblue')
plt.title("The Most Frequent U.S. Sectors")
plt.xlabel("Sector")
plt.ylabel("Freq")
plt.xticks(rotation = 45, ha = 'right')
plt.show()

#Beta
print(stock_df_v2['Beta'].describe())
high_beta = stock_df_v2[(stock_df_v2['Beta'] <= -5) | (stock_df_v2['Beta'] >= 5)]
len(high_beta) / len(stock_df_v2) # Less than 1.8% of the data affected
stock_df_v2['Beta'][stock_df_v2['Beta'] <= -5] = -5 #Set a floor at -10
stock_df_v2['Beta'][stock_df_v2['Beta'] >= 5] = 5 #Set a ceiling at 10
print(stock_df_v2['Beta'].describe())

#Market Cap
print(stock_df_v2['Market Cap'].describe())
low_market_cap = stock_df_v2[stock_df_v2['Market Cap'] <= 10000000] #Filtering small companies less than $10m
len(low_market_cap) / len(stock_df_v2) #Makes up 11% of all stocks
high_market_cap = stock_df_v2[stock_df_v2['Market Cap'] >= 100000000000] #Filtering large companies more than $100b
len(high_market_cap) / len(stock_df_v2) #Makes up 2.4% of all stocks

sns.boxplot(x = stock_df_v2['Market Cap'])
plt.title("Box Plot of Market Cap")
plt.show()

#There is a strong skew for market cap as is, going to apply a log transformation
stock_df_v2['Market Cap'][(stock_df_v2['Market Cap'] <= 1)] = np.nan
stock_df_v2['Log Market Cap'] = np.log(stock_df_v2['Market Cap'])
stock_df_v2.drop('Market Cap', axis = 1, inplace = True)

print(stock_df_v2['Log Market Cap'].describe())
sns.boxplot(x = stock_df_v2['Log Market Cap'])
plt.title("Box Plot of the Natural Log Market Cap")
plt.show()

#Price to Book
print(stock_df_v2['Price/Book'].describe())
sns.boxplot(x = stock_df_v2['Price/Book'])
plt.title("Box Plot of Price/Book")
plt.show()

low_PB = stock_df_v2[stock_df_v2['Price/Book'] < -5]
len(low_PB) / len(stock_df_v2) #Makes up 4.8% of all stocks
low_PB['Price/Book'][(low_PB['Price/Book'] <= -50)] = np.nan
lower_bound_PB = low_PB['Price/Book'].mean() #-16.75

high_PB = stock_df_v2[stock_df_v2['Price/Book'] > 13]
len(high_PB) / len(stock_df_v2) #Makes up 5.1% of all stocks
high_PB['Price/Book'][(high_PB['Price/Book'] >= 50)] = np.nan
high_bound_PB = high_PB['Price/Book'].mean() #23.16

stock_df_v2['Price/Book'][(stock_df_v2['Price/Book'] <= lower_bound_PB)] = lower_bound_PB
stock_df_v2['Price/Book'][(stock_df_v2['Price/Book'] >= high_bound_PB)] = high_bound_PB

print(stock_df_v2['Price/Book'].describe())

sns.boxplot(x = stock_df_v2['Price/Book'])
plt.title("Box Plot of Price/Book")
plt.show() #Fat tailed distribution, but fairly symmetrical after transformation

#Dividend Yield
print(stock_df_v2['DividendYield'].describe())

sns.boxplot(x = stock_df_v2['DividendYield'])
plt.title("Box Plot of Dividend Yields")
plt.show()

high_dividends = stock_df_v2[stock_df_v2['DividendYield'] >= 50]
len(high_dividends) #Since there are only three I will drop them since dividends above 50% doesn't make sense
stock_df_v2[stock_df_v2['DividendYield'] >= 50] = np.nan

sns.boxplot(x = stock_df_v2['DividendYield'])
plt.title("Box Plot of Dividend Yields")
plt.show()

stock_df_v3 = stock_df_v2

#Historical Performance Analysis
historical_prices_v4.dtypes
prices_plot = historical_prices_v4[:-2]
returns_plot = historical_returns_v4[:-2]

#Plots for seeing the distribution of returns, volatility, and Sharpe ratios for price and total returns
fig, ax = plt.subplots(1,1)
prices_plot['2000-2024 Average Returns'].hist(bins=30, alpha = 0.7, label = "Price Appreciation")
returns_plot['2000-2024 Average Returns'].hist(bins=30, alpha = 0.7, label = "Total Return")
plt.title("Distribution of Log Average Returns 2000 - 2024")
ax.set(xlabel = "% Annual Return", ylabel = "Number of Companies")
plt.legend()
plt.show()

fig, ax = plt.subplots(1,1)
prices_plot['2000-2024 Average Volatility'].hist(bins=30, alpha = 0.7, label = "Price Appreciation")
returns_plot['2000-2024 Average Volatility'].hist(bins=30, alpha = 0.7, label = "Total Return")
plt.title("Distribution of Average Volatility 2000 - 2024")
ax.set(xlabel = "% Volatility", ylabel = "Number of Companies")
plt.legend(loc = "upper right")
plt.show()

fig, ax = plt.subplots(1,1)
prices_plot['2000-2024 Sharpe Ratio'].hist(bins=30, alpha = 0.7, label = "Price Appreciation")
returns_plot['2000-2024 Sharpe Ratio'].hist(bins=30, alpha = 0.7, label = "Total Return")
plt.title("Distribution of Sharpe Ratios 2000 - 2024")
ax.set(xlabel = "Sharpe Ratio", ylabel = "Number of Companies")
plt.legend(loc = "upper right")
plt.show()

len(historical_prices_v4[historical_prices_v4['2000-2024 Sharpe Ratio'] > 0]) #3,245 companies
len(historical_returns_v4[historical_returns_v4['2000-2024 Sharpe Ratio'] > 0]) #3,761 companies
historical_prices_v4['2000-2024 Sharpe Ratio'].describe()
historical_returns_v4['2000-2024 Sharpe Ratio'].describe()

fig, ax = plt.subplots(1,1)
returns_plot['2000-2014 Sharpe Ratio'].hist(bins=30, alpha = 0.7, label ='2000-2014')
returns_plot['2015-2024 Sharpe Ratio'].hist(bins=30, alpha = 0.7, label ='2015-2024')
plt.title('Historical Period Comparison of Sharpe Ratios')
ax.set(xlabel = 'Sharpe Ratio', ylabel = 'Number of Companies')
plt.legend(loc = 'upper right')
plt.show()

fig, ax = plt.subplots(1,1)
returns_plot['2000-2014 Average Returns'].hist(bins=30, alpha = 0.7, label ='2000-2014')
returns_plot['2015-2024 Average Returns'].hist(bins=30, alpha = 0.7, label ='2015-2024')
plt.title('Historical Period Comparison of Annual Returns')
ax.set(xlabel = 'Annual Return %', ylabel = 'Number of Companies')
plt.legend(loc = 'upper right')
plt.show()

fig, ax = plt.subplots(1,1)
returns_plot['2000-2014 Average Volatility'].hist(bins=30, alpha = 0.7, label ='2000-2014')
returns_plot['2015-2024 Average Volatility'].hist(bins=30, alpha = 0.7, label ='2015-2024')
plt.title('Historical Period Comparison of Annual Volatility')
ax.set(xlabel = 'Volatility %', ylabel = 'Number of Companies')
plt.legend(loc = 'upper right')
plt.show()

#Measures of Kurtosis
def tail_count(hist_list, mean, sd):
    lower = mean - 1.645 * sd
    upper = mean + 1.645 * sd
    tail_num = 0
    for ret in hist_list:
        if ret < lower or ret > upper:
            tail_num +=1
    
    return tail_num

# Create a new feature that counts the number of times a stock price is outside the 95% confidence interval of its mean
historical_prices_v4['2000-2014 Tail_Freq'] = historical_prices_v4.apply(lambda row: tail_count(row[training_period], row['2000-2014 Average Returns'], row['2000-2014 Average Volatility']), axis = 1)
historical_prices_v4['2000-2014 Tail_Freq'].describe()

historical_returns_v4['2000-2014 Tail_Freq'] = historical_returns_v4.apply(lambda row: tail_count(row[training_period], row['2000-2014 Average Returns'], row['2000-2014 Average Volatility']), axis = 1)
historical_returns_v4['2000-2014 Tail_Freq'].describe()

#Correlational Analysis - Numerical
numerical_stock = stock_df_v3[['Ticker', 'P/E', 'Beta', 'Price/Book', 'DividendYield']]
numerical_stock.index = numerical_stock['Ticker']
price_vars = pd.merge(numerical_stock, historical_prices_v4[['2000-2024 Average Returns', '2000-2024 Average Volatility', 
                                                            '2000-2024 Sharpe Ratio', '2000-2014 Tail_Freq']], left_index = True, right_index = True)
return_vars = pd.merge(numerical_stock, historical_returns_v4[['2000-2024 Average Returns', '2000-2024 Average Volatility', 
                                                            '2000-2024 Sharpe Ratio', '2000-2014 Tail_Freq']], left_index = True, right_index = True)
corr_matrix_price = price_vars.iloc[:, 1:].corr()
print(corr_matrix_price)

fig, ax = plt.subplots(1,1)
sns.heatmap(corr_matrix_price, cmap = 'coolwarm')
plt.title('Correlation Heatmap Price Appreciation')
plt.show()

corr_matrix_return = return_vars.iloc[:, 1:].corr()
print(corr_matrix_return)

fig, ax = plt.subplots(1,1)
sns.heatmap(corr_matrix_return, cmap = 'coolwarm')
plt.title('Correlation Heatmap Total Return')
plt.show()
#Low negative correlations with each other


#Correlational Analysis - Categorical No charts for Industry since it would look unclean with the significant num of levels
cat_stock = stock_df_v3[['Ticker', 'Sector', 'SP500', 'Meme']]
cat_stock.index = cat_stock['Ticker']
cat_return = pd.merge(cat_stock, historical_returns_v4[['2000-2024 Average Returns', '2000-2024 Average Volatility', '2000-2024 Sharpe Ratio']],
                      left_index = True, right_index = True)

cat_return.drop('Ticker', axis = 1, inplace = True)

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.violinplot(x=cat_return['Sector'], y = cat_return['2000-2024 Sharpe Ratio'])
plt.xticks(fontsize = 18, rotation = 'vertical')
plt.yticks(fontsize = 18)
plt.xlabel('Sector', fontsize = 24)
plt.ylabel('Sharpe Ratio', fontsize = 24)
plt.title('Sharpe Ratio by Sector', fontsize = 36)

color_list = ['red', 'blue', 'green', 'orange', 'gold', 'purple', 'pink', 'olive', 'cyan', 'magenta', 'tan']
med_returns_cat = cat_return.groupby('Sector')['2000-2024 Average Returns'].median()
med_vol_cat = cat_return.groupby('Sector')['2000-2024 Average Volatility'].median()
med_sharpe_cat = cat_return.groupby('Sector')['2000-2024 Sharpe Ratio'].median()

fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.barh(med_returns_cat.index, med_returns_cat, color = color_list)
ax1.set_ylabel('Sector', fontsize = 14)
ax1.set_xlabel('Median Annual Return', fontsize = 12)
fig.suptitle('Comparison of Annual Returns 2000-2024 by Sector', fontsize = 18)

ax2.barh(med_vol_cat.index, med_vol_cat, color = color_list)
ax2.set_xlabel('Median Volatility', fontsize = 12)
ax2.set_yticks([])
fig.tight_layout()

fig, ax = plt.subplots(1,1)
plt.figure(figsize = (8,8))
plt.barh(med_sharpe_cat.index, med_sharpe_cat, color = color_list)
plt.ylabel('Sector')
plt.xlabel('Sharpe Ratio')
plt.title('Comparison of Sharpe Ratio 2000-2024 by Sector')

healthcare = cat_return[cat_return['Sector'] == 'Healthcare']
fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.boxplot(x = healthcare['Sector'], y = healthcare['2000-2024 Average Returns'])
plt.xticks([])
plt.yticks(fontsize = 18)
plt.xlabel('Healthcare', fontsize = 24)
plt.ylabel('Annual Return', fontsize = 24)
plt.title('Annual Return for Healthcare Sector', fontsize = 36)

healthcare['2000-2024 Average Returns'].describe() #most companies are not profitable

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.violinplot(x=cat_return['SP500'], y = cat_return['2000-2024 Sharpe Ratio'])
plt.xticks(fontsize = 18, rotation = 'vertical')
plt.yticks(fontsize = 18)
plt.xlabel('SP500', fontsize = 24)
plt.ylabel('Sharpe Ratio', fontsize = 24)
wrapped_title = textwrap.fill('Sharpe Ratio based on whether it is in the SP500', width = 30)
plt.title(wrapped_title, fontsize = 24)

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.violinplot(x=cat_return['SP500'], y = cat_return['2000-2024 Average Returns'])
plt.xticks(fontsize = 18, rotation = 'vertical')
plt.yticks(fontsize = 18)
plt.xlabel('SP500', fontsize = 24)
plt.ylabel('Annual Return', fontsize = 24)
wrapped_title = textwrap.fill('Annual Return based on whether it is in the SP500', width = 30)
plt.title(wrapped_title, fontsize = 24)

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.violinplot(x=cat_return['SP500'], y = cat_return['2000-2024 Average Volatility'])
plt.xticks(fontsize = 18, rotation = 'vertical')
plt.yticks(fontsize = 18)
plt.xlabel('SP500', fontsize = 24)
plt.ylabel('Volatility', fontsize = 24)
wrapped_title = textwrap.fill('Volatility based on whether it is in the SP500', width = 30)
plt.title(wrapped_title, fontsize = 24)

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.violinplot(x=cat_return['Meme'], y = cat_return['2000-2024 Sharpe Ratio'])
plt.xticks(fontsize = 18, rotation = 'vertical')
plt.yticks(fontsize = 18)
plt.xlabel('Meme stock', fontsize = 24)
plt.ylabel('Sharpe Ratio', fontsize = 24)
wrapped_title = textwrap.fill('Sharpe Ratio based on whether it is popular on WallStreetBets', width = 35)
plt.title(wrapped_title, fontsize = 24)

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.violinplot(x=cat_return['Meme'], y = cat_return['2000-2024 Average Returns'])
plt.xticks(fontsize = 18, rotation = 'vertical')
plt.yticks(fontsize = 18)
plt.xlabel('Meme stock', fontsize = 24)
plt.ylabel('Annual Return', fontsize = 24)
wrapped_title = textwrap.fill('Annual Return based on whether it is popular on WallStreetBets', width = 35)
plt.title(wrapped_title, fontsize = 24)

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.violinplot(x=cat_return['Meme'], y = cat_return['2000-2024 Average Volatility'])
plt.xticks(fontsize = 18, rotation = 'vertical')
plt.yticks(fontsize = 18)
plt.xlabel('Meme stock', fontsize = 24)
plt.ylabel('Volatility', fontsize = 24)
wrapped_title = textwrap.fill('Volatility based on whether it is popular on WallStreetBets', width = 35)
plt.title(wrapped_title, fontsize = 24)


### Model Building
# This section of the code performs the model building used for presentation 4
# Style Investing
stocks_cluster = stock_df_v3[(stock_df_v3['Log Market Cap'].notna()) & (stock_df_v3['P/E'].notna())]
scaler = StandardScaler()
Cap_Scaled = scaler.fit_transform(np.array(stocks_cluster['Log Market Cap']).reshape(-1, 1))
PE_Scaled = scaler.fit_transform(np.array(stocks_cluster['P/E']).reshape(-1, 1))
x = np.hstack((PE_Scaled, Cap_Scaled))
kmean_clusters = range(1, 16)
results = []

for k in kmean_clusters:
    kmeans = KMeans(n_clusters=k, random_state = 113, n_init = 10)
    kmeans.fit(x)
    results.append(kmeans.inertia_)

#Elbow Plot
fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8,8))
plt.scatter(kmean_clusters, results, c = 'black')
plt.title('Elbow Plot Kmeans Clustering Market Cap and P/E')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show() #5 clusters seems about best

kmeans = KMeans(n_clusters = 5, random_state = 113, n_init = 10)
kmeans.fit(x)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#Plot of 5 clusters
fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
plt.scatter(PE_Scaled, Cap_Scaled, c = labels, cmap = 'rainbow', s = 50, alpha = 0.8)
plt.scatter(centroids[:,0], centroids[:, 1], s=200, marker = 'x', c='black', label = 'centroids')
plt.title('Plot of Market Cap and P/E Clustering')
plt.xlabel('P/E Scaled')
plt.ylabel('Market Scaled Scaled')
plt.legend()
plt.grid(True)
plt.show()

#Need to assign cluster labels to corresponding stocks and merge stock performance to clusters
stocks_cluster['cluster'] = labels
pricereturn_mapping = historical_prices_v4['2000-2024 Average Returns']
pricevol_mapping = historical_prices_v4['2000-2024 Average Volatility']
pricesharpe_mapping = historical_prices_v4['2000-2024 Sharpe Ratio']

totalreturn_mapping = historical_returns_v4['2000-2024 Average Returns']
totalvol_mapping = historical_returns_v4['2000-2024 Average Volatility']
totalsharpe_mapping = historical_returns_v4['2000-2024 Sharpe Ratio']

stocks_cluster['PriceReturns'] = stocks_cluster['Ticker'].map(pricereturn_mapping)
stocks_cluster['PriceVol'] = stocks_cluster['Ticker'].map(pricevol_mapping)
stocks_cluster['PriceSharpe'] = stocks_cluster['Ticker'].map(pricesharpe_mapping)
stocks_cluster['TotalReturns'] = stocks_cluster['Ticker'].map(totalreturn_mapping)
stocks_cluster['TotalVol'] = stocks_cluster['Ticker'].map(totalvol_mapping)
stocks_cluster['TotalSharpe'] = stocks_cluster['Ticker'].map(totalsharpe_mapping)

#Cluster 1 - Large-Cap Investing
cluster_1 = stocks_cluster[stocks_cluster['cluster'] == 1]
cluster_1.describe() #Log Market Cap 23.72, P/E = 20.32
cluster_1['PriceReturns'].describe() #Mean = 9.81%
cluster_1['PriceVol'].describe() #Vol = 35.8%
cluster_1['PriceSharpe'].describe() #Sharpe Ratio: 0.208
cluster_1['TotalReturns'].describe() #Mean = 10.82%
cluster_1['TotalVol'].describe() #Vol = 32.2%
cluster_1['TotalSharpe'].describe() #Sharpe Ratio: 0.28


#Cluster 2 - Mid-Cap Blended Investing
cluster_2 = stocks_cluster[stocks_cluster['cluster'] == 2]
cluster_2.describe() #Log Market Cap 20.59, P/E = 9.25
cluster_2['PriceReturns'].describe() #Mean = -3.5%
cluster_2['PriceVol'].describe() #Vol = 51.8%
cluster_2['PriceSharpe'].describe() #Sharpe Ratio: -0.117
cluster_2['TotalReturns'].describe() #Mean = -1.95%
cluster_2['TotalVol'].describe() #Vol = 47.6%
cluster_2['TotalSharpe'].describe() #Sharpe Ratio: -0.04


#Cluster 3 - Small Cap
cluster_3 = stocks_cluster[stocks_cluster['cluster'] == 3]
cluster_3.describe() #Log Market Cap 17.2, P/E = -2.17
cluster_3['PriceReturns'].describe() #Mean = -44.87%
cluster_3['PriceVol'].describe() #Vol = 78.7%
cluster_3['PriceSharpe'].describe() #Sharpe Ratio: -0.705
cluster_3['TotalReturns'].describe() #Mean = -43.55%
cluster_3['TotalVol'].describe() #Vol = 76.6%
cluster_3['TotalSharpe'].describe() #Sharpe Ratio: -0.68


#Cluster 4 Growth Investing
cluster_4 = stocks_cluster[stocks_cluster['cluster'] == 4]
cluster_4.describe() #Log Market Cap 21.41, P/E = 81.55
cluster_4['PriceReturns'].describe() #Mean = -1.81%
cluster_4['PriceVol'].describe() #Vol = 61%
cluster_4['PriceSharpe'].describe() #Sharpe Ratio: -0.065
cluster_4['TotalReturns'].describe() #Mean = -0.85%
cluster_4['TotalVol'].describe() #Vol = 59.1%
cluster_4['TotalSharpe'].describe() #Sharpe Ratio: -0.012

# Cluster 5 Value or Contrarian Investing
cluster_5 = stocks_cluster[stocks_cluster['cluster'] == 0]
cluster_5.describe() #Log Market Cap 20.5, P/E = -71.4
cluster_5['PriceReturns'].describe() #Mean = -9.38%
cluster_5['PriceVol'].describe() #Vol = 71%
cluster_5['PriceSharpe'].describe() #Sharpe Ratio: -0.179
cluster_5['TotalReturns'].describe() #Mean = -8.58%
cluster_5['TotalVol'].describe() #Vol = 68.9%
cluster_5['TotalSharpe'].describe() #Sharpe Ratio: -0.139

#Summarizing Results for the Presentation in table format
Cluster_list = ['Large Cap (Blue)', 'Mid Cap (Green)', 'Small Cap (Orange)', 'Growth (Red)', 'Value/Contrarian (Purple)']
Cluster_PE = [cluster_1['P/E'].mean().round(1), cluster_2['P/E'].mean().round(1), 
              cluster_3['P/E'].mean().round(1), cluster_4['P/E'].mean().round(1), cluster_5['P/E'].mean().round(1)]
Cluster_Market = [cluster_1['Log Market Cap'].mean().round(1), cluster_2['Log Market Cap'].mean().round(1), cluster_3['Log Market Cap'].mean().round(1),
                cluster_4['Log Market Cap'].mean().round(1), cluster_5['Log Market Cap'].mean().round(1)]
Price_Sharpe = [cluster_1['PriceSharpe'].mean().round(3), cluster_2['PriceSharpe'].mean().round(3), cluster_3['PriceSharpe'].mean().round(3),
                cluster_4['PriceSharpe'].mean().round(3), cluster_5['PriceSharpe'].mean().round(3)]
Total_Sharpe = [cluster_1['TotalSharpe'].mean().round(3), cluster_2['TotalSharpe'].mean().round(3), cluster_3['TotalSharpe'].mean().round(3),
                cluster_4['TotalSharpe'].mean().round(3), cluster_5['TotalSharpe'].mean().round(3)]

cluster_df = pd.DataFrame({'Cluster': Cluster_list, 'P/E Center': Cluster_PE, 'Market Cap Center': Cluster_Market, 'Price Sharpe Ratio': Price_Sharpe, 'Total Sharpe Ratio': Total_Sharpe})

fig, ax = plt.subplots(figsize = (18, 6))
ax.axis('off')
table = ax.table(cellText = cluster_df.values, colLabels = cluster_df.columns, loc = 'center', colWidths = [0.2, 0.1, 0.1, 0.1, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(2, 4)
plt.title("Cluster Results", fontsize = 32)


## Predictive Model - Models:GLM, lasso regression, ridge regression, decision trees, boosted trees
stock_final = stock_df_v3
stock_final.index = stock_final['Ticker']
stock_final['2000-2014 Price Return'] = historical_prices_v4['2000-2014 Average Returns']
stock_final['2000-2014 Price Volatility'] = historical_prices_v4['2000-2014 Average Volatility']
stock_final['2000-2014 Price Sharpe Ratio'] = historical_prices_v4['2000-2014 Sharpe Ratio']
stock_final['2015-2024 Price Return'] = historical_prices_v4['2015-2024 Average Returns']
stock_final['2015-2024 Price Volatility'] = historical_prices_v4['2015-2024 Average Volatility']
stock_final['2015-2024 Price Sharpe Ratio'] = historical_prices_v4['2015-2024 Sharpe Ratio']
stock_final['2000-2014 Total Return'] = historical_returns_v4['2000-2014 Average Returns']
stock_final['2000-2014 Total Volatility'] = historical_returns_v4['2000-2014 Average Volatility']
stock_final['2000-2014 Total Sharpe Ratio'] = historical_returns_v4['2000-2014 Sharpe Ratio']
stock_final['2015-2024 Total Return'] = historical_returns_v4['2015-2024 Average Returns']
stock_final['2015-2024 Total Volatility'] = historical_returns_v4['2015-2024 Average Volatility']
stock_final['2015-2024 Total Sharpe Ratio'] = historical_returns_v4['2015-2024 Sharpe Ratio']

stock_final = stock_final.dropna(subset = ['2015-2024 Price Sharpe Ratio'])
#Set all NA's to median or unique string for categorical variables
stock_final['P/E'] = stock_final['P/E'].fillna(stock_final['P/E'].median())
stock_final['Industry'] = stock_final['Industry'].fillna('unknown')
stock_final['sector'] = stock_final['Sector'].fillna('unknown')
stock_final['Beta'] = stock_final['Beta'].fillna(stock_final['Beta'].median())
stock_final['Price/Book'] = stock_final['Price/Book'].fillna(stock_final['Price/Book'].median())
stock_final['DividendYield'] = stock_final['DividendYield'].fillna(stock_final['DividendYield'].median())
stock_final['Log Market Cap'] = stock_final['Log Market Cap'].fillna(stock_final['Log Market Cap'].median())
stock_final['2000-2014 Price Return'] = stock_final['2000-2014 Price Return'].fillna(stock_final['2000-2014 Price Return'].median())
stock_final['2000-2014 Price Volatility'] = stock_final['2000-2014 Price Volatility'].fillna(stock_final['2000-2014 Price Volatility'].median())
stock_final['2000-2014 Total Return'] = stock_final['2000-2014 Total Return'].fillna(stock_final['2000-2014 Total Return'].median())
stock_final['2000-2014 Total Volatility'] = stock_final['2000-2014 Total Volatility'].fillna(stock_final['2000-2014 Total Volatility'].median())
stock_final = pd.get_dummies(stock_final, columns = ['Industry'], drop_first = True)
stock_final = pd.get_dummies(stock_final, columns = ['Sector'], drop_first = True)
stock_final = stock_final.replace({True : 1, False: 0})
## Price Apprecation Model

# Initialization
response = stock_final['2015-2024 Price Sharpe Ratio']
features = stock_final.iloc[:, np.r_[2:11, 22:174]]
num_folds = 5
kfolds = KFold(n_splits = num_folds, shuffle = True, random_state = 113)
best_score_glm = 999
best_score_lasso = 999
best_score_ridge = 999
best_score_tree = 999
best_score_boost = 999
best_alpha_lasso = 0.0
best_alpha_ridge = 0.0

#The following modifcation to the glm library is needed in order to perform cross validation on glm
class GLMEstimator(BaseEstimator, RegressorMixin):
        def __init__(self, family=sm.families.Gaussian(), link=None):
            self.family = family
            self.link = link

        def fit(self, X, y):
            self.model = sm.GLM(y, sm.add_constant(X), family=self.family, link=self.link)
            self.results = self.model.fit()
            return self

        def predict(self, X):
            return self.results.predict(sm.add_constant(X))

test = sm.GLM(response, features, family = sm.families.Gaussian()).fit()
model_glm = GLMEstimator(family = sm.families.Gaussian())
best_score_glm = -np.mean(cross_val_score(model_glm, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))

for alpha_value in np.arange(0.0, 1.2, 0.2):
    model_lasso = Lasso(alpha = alpha_value)
    score_lasso = -np.mean(cross_val_score(model_lasso, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))
    if score_lasso < best_score_lasso:
        best_score_lasso = score_lasso
        best_alpha_lasso = alpha_value
        
    model_ridge = Ridge(alpha = alpha_value)
    score_ridge = -np.mean(cross_val_score(model_ridge, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))
    if score_ridge < best_score_ridge:
        best_score_ridge = score_ridge
        best_alpha_ridge = alpha_value

tree_params = {
    'max_depth': [5,10,20],
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [1,2,5]
    }

tuning_tree = RandomizedSearchCV(estimator = DecisionTreeRegressor(),param_distributions = tree_params)
tuning_tree.fit(features, response)
model_tree = DecisionTreeRegressor(**tuning_tree.best_params_, random_state = 113)
best_score_tree = -np.mean(cross_val_score(model_tree, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))

model_boost_params = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [2, 3, 4],
    'subsample': [0.5, 0.65, 0.8]
    }



tuning_boost = RandomizedSearchCV(estimator = GradientBoostingRegressor(),param_distributions = model_boost_params)
tuning_boost.fit(features, response)
model_boost = GradientBoostingRegressor(**tuning_boost.best_params_, random_state = 113)
best_score_boost = -np.mean(cross_val_score(model_boost, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))

price_model_results = [
    ["GLM", round(best_score_glm, 4), "N/A"],
    ["Lasso", round(best_score_lasso,4), best_alpha_lasso],
    ["Ridge", round(best_score_ridge,4), best_alpha_ridge],
    ["Decision Tree", round(best_score_tree, 4), tuning_tree.best_params_],
    ["Boosted Tree", round(best_score_boost, 4), tuning_boost.best_params_]
    ]
headers = ['Model', 'MSE', 'Best Parameters']

price_df = pd.DataFrame(price_model_results)
price_df.columns = headers
fig, ax = plt.subplots(figsize = (16, 6))
ax.axis('off')
table = ax.table(cellText = price_df.values, colLabels = price_df.columns, loc = 'center', colWidths = [0.1, 0.1, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(2, 3)
plt.title("Stock Price Appreciation Model-Building Cross Validation Results", fontsize = 32)

#Are there any data points with outsized value (outliers, highly leveraged points)

final_model_price = model_boost
final_model_price.fit(features, response)
predicted_sharpe_price = pd.Series(final_model_price.predict(features), name = 'Price Predictions', index = features.index)
price_results = pd.concat([response, predicted_sharpe_price], axis = 1)
price_results['Error'] = price_results['2015-2024 Price Sharpe Ratio'] - price_results['Price Predictions']
price_results['Error'].describe()

large_price_errors = price_results[(price_results['Error'] < -1) | (price_results['Error'] > 1)]
large_price_errors = pd.merge(large_price_errors, features, left_index = True, right_index = True, how = 'left')
large_price_errors.dtypes
large_price_errors['P/E'].describe() #Extreme or missing
large_price_errors['Beta'].describe() #Nothing noteworthy
large_price_errors['Price/Book'].describe() #Nothing noteworthy
large_price_errors['DividendYield'].describe() #Extreme or missing
large_price_errors['SP500'].describe() #Only 4 companies, lower portion overall
large_price_errors['Meme'].describe() #Only 1 company, lower portion overall
large_price_errors['Log Market Cap'].describe() #Wide variety, mean especially close to median
large_price_errors['2000-2014 Price Return'].describe() #Extreme or missing
large_price_errors['2000-2014 Price Volatility'].describe() #Mostly missing

#Feature importance
importance_price = pd.DataFrame({'Feature': features.columns, 'Importance': final_model_price.feature_importances_})
importance_price = importance_price.sort_values(by = 'Importance', ascending = False)
imp_price = importance_price.head(10)
imp_price['Importance'] = imp_price['Importance'].round(3)

fig, ax = plt.subplots(figsize = (2, 8))
ax.axis('off')
table = ax.table(cellText = imp_price.values, colLabels = imp_price.columns, loc = 'center', colWidths = [1.5, 1])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(2, 3)
plt.title("Top 10 Most Important Features - Price Return Predictive Model", fontsize = 32)

r_squared = r2_score(price_results['2015-2024 Price Sharpe Ratio'], price_results['Price Predictions'])
r_squared #0.3756 - suggestive that the model is not powerful in predictive ability and room for improvement

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.residplot(x=price_results['Price Predictions'], y = price_results['2015-2024 Price Sharpe Ratio'], lowess = True, line_kws = {'color': 'black'})
plt.xlabel('Predicted Price Sharpe Ratio')
plt.ylabel('Actual Price Sharpe Ratio')
plt.title("Predicted vs. Actual Price Sharpe Ratio 2015-2024")
plt.show()

# Top 100 stocks based on the Price Predictive Model
price_results = price_results.sort_values(by = 'Price Predictions', ascending = False)
price_model_portfolio = price_results.head(100)
price_model_portfolio


## Total Return model building

# Initialization
response = stock_final['2015-2024 Total Sharpe Ratio']
features = stock_final.iloc[:, np.r_[2:9, 15:17, 22:174]]
num_folds = 5
kfolds = KFold(n_splits = num_folds, shuffle = True, random_state = 113)
best_score_glm = 999
best_score_lasso = 999
best_score_ridge = 999
best_score_tree = 999
best_score_boost = 999
best_alpha_lasso = 0.0
best_alpha_ridge = 0.0

#Building the models
cross_val_score(model_glm, features, response, cv = kfolds, scoring = 'neg_mean_squared_error')

model_glm = GLMEstimator(family = sm.families.Gaussian())
best_score_glm = -np.mean(cross_val_score(model_glm, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))

for alpha_value in np.arange(0, 1.2, 0.2):
    model_lasso = Lasso(alpha = alpha_value)
    score_lasso = -np.mean(cross_val_score(model_lasso, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))
    if score_lasso < best_score_lasso:
        best_score_lasso = score_lasso
        best_alpha_lasso = alpha_value
        
    model_ridge = Ridge(alpha = alpha_value)
    score_ridge = -np.mean(cross_val_score(model_ridge, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))
    if score_ridge < best_score_ridge:
        best_score_ridge = score_ridge
        best_alpha_ridge = alpha_value

tree_params = {
    'max_depth': [5,10,20],
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [1,2,5]
    }

tuning_tree = RandomizedSearchCV(estimator = DecisionTreeRegressor(),param_distributions = tree_params)
tuning_tree.fit(features, response)
model_tree = DecisionTreeRegressor(**tuning_tree.best_params_, random_state = 113)
best_score_tree = -np.mean(cross_val_score(model_tree, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))

model_boost_params = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [2, 3, 4],
    'subsample': [0.5, 0.65, 0.8]
    }

tuning_boost = RandomizedSearchCV(estimator = GradientBoostingRegressor(),param_distributions = model_boost_params)
tuning_boost.fit(features, response)
model_boost = GradientBoostingRegressor(**tuning_boost.best_params_, random_state = 113)
best_score_boost = -np.mean(cross_val_score(model_boost, features, response, cv = kfolds, scoring = 'neg_mean_squared_error'))

return_model_results = [
    ["GLM", round(best_score_glm,4), "N/A"],
    ["Lasso", round(best_score_lasso,4), best_alpha_lasso],
    ["Ridge", round(best_score_ridge,4), best_alpha_ridge],
    ["Decision Tree", round(best_score_tree,4), tuning_tree.best_params_],
    ["Boosted Tree", round(best_score_boost,4), tuning_boost.best_params_]
    ]
headers = ['Model', 'MSE', 'Best Parameters']

return_df = pd.DataFrame(return_model_results)
return_df.columns = headers
fig, ax = plt.subplots(figsize = (16, 6))
ax.axis('off')
table = ax.table(cellText = return_df.values, colLabels = return_df.columns, loc = 'center', colWidths = [0.1, 0.1, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(2, 3)
plt.title("Total Return Model-Building Cross Validation Results", fontsize = 32)

#Are there any data points with outsized value (outliers, highly leveraged points)

final_model_return = model_boost
final_model_return.fit(features, response)
predicted_sharpe_return = pd.Series(final_model_return.predict(features), name = 'Return Predictions', index = features.index)
return_results = pd.concat([response, predicted_sharpe_return], axis = 1)
return_results['Error'] = return_results['2015-2024 Total Sharpe Ratio'] - return_results['Return Predictions']
return_results['Error'].describe()

large_return_errors = return_results[(return_results['Error'] < -1) | (return_results['Error'] > 1)]
large_return_errors = pd.merge(large_return_errors, features, left_index = True, right_index = True, how = 'left')

large_return_errors['P/E'].describe() #Extreme or missing
large_return_errors['Beta'].describe() #Nothing noteworthy
large_return_errors['Price/Book'].describe() #Nothing noteworthy
large_return_errors['DividendYield'].describe() #Extreme or missing
large_return_errors['SP500'].describe() #Only 5 companies, lower portion overall
large_return_errors['Meme'].describe() #Only 1 company, lower portion overall
large_return_errors['Log Market Cap'].describe() #Wide variety, mean especially close to median
large_return_errors['2000-2014 Total Return'].describe() #Extreme or missing
large_return_errors['2000-2014 Total Volatility'].describe() #Mostly missing


#Feature importance
importance_return = pd.DataFrame({'Feature': features.columns, 'Importance': final_model_return.feature_importances_})
importance_return = importance_return.sort_values(by = 'Importance', ascending = False)
imp_return = importance_return.head(10)
imp_return['Importance'] = imp_return['Importance'].round(3)

fig, ax = plt.subplots(figsize = (2, 8))
ax.axis('off')
table = ax.table(cellText = imp_return.values, colLabels = imp_return.columns, loc = 'center', colWidths = [1.5, 1])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(2, 3)
plt.title("Top 10 Most Important Features - Total Return Predictive Model", fontsize = 32)

r_squared = r2_score(return_results['2015-2024 Total Sharpe Ratio'], return_results['Return Predictions'])
r_squared #0.445

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(8, 8))
sns.residplot(x=return_results['Return Predictions'], y = return_results['2015-2024 Total Sharpe Ratio'], lowess = True, line_kws = {'color': 'black'})
plt.xlabel('Predicted Return Sharpe Ratio')
plt.ylabel('Actual Return Sharpe Ratio')
plt.title("Predicted vs. Actual Total Return Sharpe Ratio 2015-2024")
plt.show()

#The remaining portion of the code is support and analysis for the final results in the last presentation and paper

# Top 100 stocks based on the Total return Predictive Model
return_results = return_results.sort_values(by = 'Return Predictions', ascending = False)
return_model_portfolio = return_results.head(100)
return_model_portfolio


## Other strategies to analyze

# High Dividend Yields - Only include companies that offer dividends greater than 5%
high_dividends = stock_final[stock_final['DividendYield'] >= 5]


# Volatility Controlled - include 2 strategies, include stocks with volatility less than 20% and one less than 40% between 2000-2014 on price volatility only
vol20 = stock_final[stock_final['2000-2014 Price Volatility'] <= 20]

vol40 = stock_final[stock_final['2000-2014 Price Volatility'] <= 40]

#Prep the final datasets for analysis to answer the project objectives
final_results = stock_final[['Company Name', 'Meme', 'SP500', '2015-2024 Price Return', '2015-2024 Price Volatility', 
                             '2015-2024 Price Sharpe Ratio', '2015-2024 Total Return', '2015-2024 Total Volatility', 
                             '2015-2024 Total Sharpe Ratio']]
final_results['Price-Model'] = final_results.index.isin(price_model_portfolio.index).astype(int)
final_results['Return-Model'] = final_results.index.isin(return_model_portfolio.index).astype(int)
final_results['Style-LargeCap'] = final_results.index.isin(cluster_1['Ticker']).astype(int)
final_results['Style-MidCap'] = final_results.index.isin(cluster_2['Ticker']).astype(int)
final_results['Style-SmallCap'] = final_results.index.isin(cluster_3['Ticker']).astype(int)
final_results['Style-Growth'] = final_results.index.isin(cluster_4['Ticker']).astype(int)
final_results['Style-Value'] = final_results.index.isin(cluster_5['Ticker']).astype(int)
final_results['High-Dividends'] = final_results.index.isin(high_dividends.index).astype(int)
final_results['Vol_Control-20'] = final_results.index.isin(vol20.index).astype(int)
final_results['Vol_Control-40'] = final_results.index.isin(vol40.index).astype(int)


#Question 2 - How much value does a total return portfolio provide overall?
final_results[['2015-2024 Price Sharpe Ratio', '2015-2024 Total Sharpe Ratio']].describe()
final_results['2015-2024 Total Sharpe Ratio'].mean() - final_results['2015-2024 Price Sharpe Ratio'].mean() #0.069 increase
final_results['2015-2024 Total Sharpe Ratio'].median() - final_results['2015-2024 Price Sharpe Ratio'].median() #0.0484 increase
f_statistic, p_value = stats.f_oneway(final_results['2015-2024 Price Sharpe Ratio'], final_results['2015-2024 Total Sharpe Ratio']) #Volatilities are statistically different
t_statistic, p_value = stats.ttest_ind(final_results['2015-2024 Total Sharpe Ratio'], final_results['2015-2024 Price Sharpe Ratio'], equal_var=False, alternative = 'greater')
Total_No_Outliers = final_results[(final_results['2015-2024 Total Sharpe Ratio'] >= -1.5)
                                                & (final_results['2015-2024 Total Sharpe Ratio'] <= 1.5)]
Price_No_Outliers = final_results[(final_results['2015-2024 Price Sharpe Ratio'] >= -1.5)
                                                & (final_results['2015-2024 Price Sharpe Ratio'] <= 1.5)]

fig, axes = plt.subplots(1,2, figsize = (30, 25))

sns.violinplot(y = Total_No_Outliers['2015-2024 Total Sharpe Ratio'], ax = axes[0], inner = 'quart')
for i, line in enumerate(axes[0].lines):
    line.set_linewidth(5)
axes[0].set_title('Sharpe Ratio Distribution of Total Returns', fontsize = 40)
axes[0].set_ylabel("Sharpe Ratio", fontsize = 32)
axes[0].tick_params(axis = 'y', labelsize = 24)
sns.violinplot(y = Price_No_Outliers['2015-2024 Price Sharpe Ratio'], ax = axes[1], inner = 'quart')
for i, line in enumerate(axes[1].lines):
    line.set_linewidth(5)
axes[1].set_title('Sharpe Ratio Distribution of Price-only Returns', fontsize = 40)
axes[1].set_ylabel("Sharpe Ratio", fontsize = 0)
axes[1].tick_params(axis = 'y', labelsize = 24)
plt.tight_layout()

#Marginal Increase in performance, no overall change in distribution adds 0.04 to 0.07 to Sharpe Ratio for increased return/volatility stability

#Question 1 - Which were the three best performing funds on a risk-adjusted return basis (also which strategies performed poorly)
portfolio_statistics = ['Sharpe Ratio', 'Average Return', 'Average Volatility']
portfolio_strategies = ['Meme', 'Meme_div', 'SP500', 'SP500_div', 'High-Yield Dividends', 'VolControl-20', 
                        'VolControl-20_div', 'VolControl-40', 'VolControl-40_div', 'LargeCap-Style', 
                        'LargeCap-Style_div', 'MidCap-Style', 'MidCap-Style_div', 'SmallCap-Style', 
                        'SmallCap-Style_div', 'Growth-Style', 'Growth-Style_div', 'Value-Style', 'Value-Style_div', 
                        'Price-Model', 'Return-Model']
portfolio_results = pd.DataFrame(np.nan, index = portfolio_statistics, columns = portfolio_strategies)

#Filling out the table - a more manual process since certain columns require certain statistics

portfolio_results.at['Sharpe Ratio', 'Meme'] = final_results[final_results['Meme'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'Meme'] = final_results[final_results['Meme'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'Meme'] = final_results[final_results['Meme'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'Meme_div'] = final_results[final_results['Meme'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'Meme_div'] = final_results[final_results['Meme'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'Meme_div'] = final_results[final_results['Meme'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'SP500'] = final_results[final_results['SP500'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'SP500'] = final_results[final_results['SP500'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'SP500'] = final_results[final_results['SP500'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'SP500_div'] = final_results[final_results['SP500'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'SP500_div'] = final_results[final_results['SP500'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'SP500_div'] = final_results[final_results['SP500'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'High-Yield Dividends'] = final_results[final_results['High-Dividends'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'High-Yield Dividends'] = final_results[final_results['High-Dividends'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'High-Yield Dividends'] = final_results[final_results['High-Dividends'] == 1]['2015-2024 Price Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'VolControl-20'] = final_results[final_results['Vol_Control-20'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'VolControl-20'] = final_results[final_results['Vol_Control-20'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'VolControl-20'] = final_results[final_results['Vol_Control-20'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'VolControl-20_div'] = final_results[final_results['Vol_Control-20'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'VolControl-20_div'] = final_results[final_results['Vol_Control-20'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'VolControl-20_div'] = final_results[final_results['Vol_Control-20'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'VolControl-40'] = final_results[final_results['Vol_Control-40'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'VolControl-40'] = final_results[final_results['Vol_Control-40'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'VolControl-40'] = final_results[final_results['Vol_Control-40'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'VolControl-40_div'] = final_results[final_results['Vol_Control-40'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'VolControl-40_div'] = final_results[final_results['Vol_Control-40'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'VolControl-40_div'] = final_results[final_results['Vol_Control-40'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'LargeCap-Style'] = final_results[final_results['Style-LargeCap'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'LargeCap-Style'] = final_results[final_results['Style-LargeCap'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'LargeCap-Style'] = final_results[final_results['Style-LargeCap'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'LargeCap-Style_div'] = final_results[final_results['Style-LargeCap'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'LargeCap-Style_div'] = final_results[final_results['Style-LargeCap'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'LargeCap-Style_div'] = final_results[final_results['Style-LargeCap'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'MidCap-Style'] = final_results[final_results['Style-MidCap'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'MidCap-Style'] = final_results[final_results['Style-MidCap'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'MidCap-Style'] = final_results[final_results['Style-MidCap'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'MidCap-Style_div'] = final_results[final_results['Style-MidCap'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'MidCap-Style_div'] = final_results[final_results['Style-MidCap'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'MidCap-Style_div'] = final_results[final_results['Style-MidCap'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'SmallCap-Style'] = final_results[final_results['Style-SmallCap'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'SmallCap-Style'] = final_results[final_results['Style-SmallCap'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'SmallCap-Style'] = final_results[final_results['Style-SmallCap'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'SmallCap-Style_div'] = final_results[final_results['Style-SmallCap'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'SmallCap-Style_div'] = final_results[final_results['Style-SmallCap'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'SmallCap-Style_div'] = final_results[final_results['Style-SmallCap'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'Growth-Style'] = final_results[final_results['Style-Growth'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'Growth-Style'] = final_results[final_results['Style-Growth'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'Growth-Style'] = final_results[final_results['Style-Growth'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'Growth-Style_div'] = final_results[final_results['Style-Growth'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'Growth-Style_div'] = final_results[final_results['Style-Growth'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'Growth-Style_div'] = final_results[final_results['Style-Growth'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'Value-Style'] = final_results[final_results['Style-Value'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'Value-Style'] = final_results[final_results['Style-Value'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'Value-Style'] = final_results[final_results['Style-Value'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'Value-Style_div'] = final_results[final_results['Style-Value'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'Value-Style_div'] = final_results[final_results['Style-Value'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'Value-Style_div'] = final_results[final_results['Style-Value'] == 1]['2015-2024 Total Volatility'].mean()

portfolio_results.at['Sharpe Ratio', 'Price-Model'] = final_results[final_results['Price-Model'] == 1]['2015-2024 Price Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'Price-Model'] = final_results[final_results['Price-Model'] == 1]['2015-2024 Price Return'].mean()
portfolio_results.at['Average Volatility', 'Price-Model'] = final_results[final_results['Price-Model'] == 1]['2015-2024 Price Volatility'].mean()
portfolio_results.at['Sharpe Ratio', 'Return-Model'] = final_results[final_results['Return-Model'] == 1]['2015-2024 Total Sharpe Ratio'].mean()
portfolio_results.at['Average Return', 'Return-Model'] = final_results[final_results['Return-Model'] == 1]['2015-2024 Total Return'].mean()
portfolio_results.at['Average Volatility', 'Return-Model'] = final_results[final_results['Return-Model'] == 1]['2015-2024 Total Volatility'].mean()

portfolio = portfolio_results.transpose()
portfolio.sort_values(by='Sharpe Ratio', ascending = False, inplace = True)
portfolio = portfolio.round(2)
portfolio['Strategy'] = portfolio.index
portfolio_ranking = portfolio[['Strategy', 'Sharpe Ratio', 'Average Return', 'Average Volatility']]

fig, ax = plt.subplots(figsize = (8, 12))
ax.axis('off')
table = ax.table(cellText = portfolio_ranking.values, colLabels = portfolio_ranking.columns, loc = 'center', colWidths = [0.4, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(2, 2)
plt.title("Investment Strategy Performance 2015-2024", fontsize = 32)

#Small stocks and or meme stocks do not do well. Sector and industry of the stock also seems to generally add trivial value

#Question 3 - What is the best estimate of long term stock performance of a U.S. stock?
historical_averages_price = historical_prices_v4
historical_averages_price = historical_averages_price[['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
                           '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']]
historical_averages = historical_averages_price.mean()
price_mean = historical_averages.mean() #-1.2%
price_volatility = historical_averages.std() #22.03%

(price_mean - 4) / price_volatility # ~ -0.23

historical_medians = historical_averages_price.median()
median_average = historical_medians.mean() #4.55%
median_volatility = historical_medians.std() #17.9%
(median_average - 4) / median_volatility # ~ 0.03

fig, ax = plt.subplots(1,1)
plt.figure(figsize=(16, 8))
plt.plot(historical_averages.index, historical_averages.values, label = 'mean')
plt.plot(historical_medians.index, historical_medians.values, label = 'median')
plt.xlabel('Year', fontsize = 24)
plt.ylabel('Return', fontsize = 24)
plt.title("Annual Returns from 2000-2024", fontsize = 32)
plt.xticks(rotation=90)
plt.legend()
plt.show()



