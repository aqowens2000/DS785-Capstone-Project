#Import Packages
import yfinance as yf 
import pandas as pd 
import datetime as dt
import warnings
import json
from datetime import timedelta
warnings.filterwarnings('ignore')

#Open the Ticker Data from the SEC
with open('company_tickers.json') as file:
    all_companies_dict = json.load(file)
all_companies = list(all_companies_dict.values())
tickers = list()
for company in all_companies:
    tickers.append(company['ticker'])
    
#Defines a function that obtains the last adjusted price in the calendar year
def get_adjusted_stock_price(ticker, start_date, end_date, cumul_div):
    current_year = start_date.year
    stock_data = yf.Ticker(ticker)
    close = stock_data.history(start = start_date, end = end_date)['Close'][-1]
    total_dividends = cumul_div + sum(stock_data.history(start = dt.datetime(current_year, 1, 1), end = dt.datetime(current_year, 12, 31))['Dividends'])
    total_return = close + total_dividends
    return close, total_return, total_dividends

#Sets the dates of the period of observation for stock performance
Last_Date = dt.datetime(2024, 12, 24)
End_Date = dt.datetime(2025, 1, 1)
Begin_Date = dt.datetime(1999, 12, 24)
Start_Date = dt.datetime(2000, 1, 1)

#Initialize the demographic queries
Stock_list = tickers
Name_list = []
PE_list = []
Industry_list = []
Sector_list = []
Beta_list = []
MarketCap_list = []
PriceBook_list = []
FirstPrice_list = []
LastPrice_list = []
DividendYield_list = []

#Query the data from the Yahoo Finance API
for stock in Stock_list:
    stock_data = yf.Ticker(stock)
    
    try:
        FirstPrice = get_adjusted_stock_price(stock, Begin_Date, Start_Date, 0)[0]
    except:
        FirstPrice = None #Stock isn't publically traded as of 1/1/2025

    try:
        LastPrice = get_adjusted_stock_price(stock, Last_Date, End_Date, 0)[0]
    except:
        LastPrice = None #Stock isn't publically traded as of 1/1/2025
    
    if FirstPrice is None and LastPrice is None:
        Name = None
        PE = None
        Industry = None
        Sector = None
        Beta = None
        MarketCap = None
        PriceBook = None
        DividendYield = None
    else:
        try:
            Name = stock_data.info['shortName']
        except:
            Name = None
            
        try:
            PE = stock_data.info['forwardPE']
        except:
            PE = None
            
        try:  
            Industry = stock_data.info['industry']
        except:
            Industry = None
    
        try:
            Sector = stock_data.info['sector']
        except:
            Sector = None
    
        try:
            Beta = stock_data.info['beta']
        except:
            Beta = None
    
        try:
            MarketCap = stock_data.info['marketCap']
        except:
            MarketCap = None
    
        try:
            PriceBook = stock_data.info['priceToBook']
        except:
            PriceBook = None
        
        try:
            DividendYield = stock_data.info['dividendYield']
        except:
            DividendYield = None
        
    Name_list.append(Name)
    PE_list.append(PE)
    Industry_list.append(Industry)
    Sector_list.append(Sector)
    Beta_list.append(Beta)
    MarketCap_list.append(MarketCap)
    PriceBook_list.append(PriceBook)
    FirstPrice_list.append(FirstPrice)
    LastPrice_list.append(LastPrice)
    DividendYield_list.append(DividendYield)

#Organize the entry into a dictionary and then a dataframe    
stock_dict = {"Ticker": Stock_list, "Company Name": Name_list, "P/E": PE_list, "Industry": Industry_list, "Sector": Sector_list, "Beta": Beta_list, "Market Cap": MarketCap_list, "Price/Book": PriceBook_list, "DividendYield": DividendYield_list, "Start Share Price": FirstPrice_list, "End Share Price": LastPrice_list}
stock_df = pd.DataFrame(stock_dict)

#Clean and Filter Tickers that are not needed to make the historical performance query quicker
stocks_clean = stock_df[(stock_df['Company Name'].notna()) & (stock_df['Start Share Price'].notna()) | (stock_df['End Share Price'].notna())]
stocks_final = stocks_clean.iloc[:, :-2]
stocks_final = stocks_final.set_index('Ticker')

#Initialize the Historical Stock Performance Query
years = []
dates = []
cal_dates = []

years = range(1999, 2025)
for year in years:
    cal_dates.append("12/31/" + str(year))
    dates.append(dt.datetime(year, 12, 24))

final_tickers = stocks_clean['Ticker'].tolist()
stock_history = []
stock_return_history = []
stock_history_dict = {}
stock_return_history_dict = {}
#Loop through each query and grab the historical data from 2000 - 2024
for ticker in final_tickers:
    cumul_div = 0
    for date in dates:
        try:
            share_price, share_return, cumul_div = get_adjusted_stock_price(ticker, date, date + timedelta(7), cumul_div) #This looks up the last price of the year
        except: #If the data isn't available set to None
            share_price = None
            share_return = None
        stock_history.append(share_price)
        stock_return_history.append(share_return)
    stock_history_dict[ticker] = stock_history
    stock_return_history_dict[ticker] = stock_return_history
    stock_history = []
    stock_return_history = []

#Save the historical prices and returns as data frames
historical_prices_df = pd.DataFrame(stock_history_dict)
historical_prices_df.index = cal_dates
historical_prices_final = historical_prices_df.transpose()

historical_returns_df = pd.DataFrame(stock_return_history_dict)
historical_returns_df.index = cal_dates
historical_returns_final = historical_returns_df.transpose()

#Export the dataframes to CSV
stocks_final.to_csv("Current Stock Data.csv", index = True)
historical_prices_final.to_csv("Historical Stock Prices.csv", index = True)
historical_returns_final.to_csv("Historical Stock Returns.csv", index = True)