# Contructing a portfolio
<details>
  <summary>Tests</summary>

  ##### 1:1 - pilot
  ```python
    #imports
import yfinance as yf
yf.pdr_override()
import pandas_datareader as pdr
from datetime import date
  ```
  ```python
class Portfolio:
    def __init__(self, tickers, start_date='2007-1-1', end_date=date.today()):
        self.raw = pdr.data.get_data_yahoo(tickers, start=start_date,end=end_date)
        self.close = self.raw['Adj Close']
        self.S = self.close[-1]
        self.stdev = self.close.std()*250**0.5
  ```
    
  ##### 2:1 - Class - default cases
In this version, defaults were added to generate a quick test case.<br> 
>A dictionary was added to access each block of data at column index level 0, but this was unnecessary as the same subsets can be simply obtained by calling the column indexes. Cool tangent, checkpointing it.
  ```python
    #imports
import yfinance as yf
import numpy as np
import datetime as dt
  ```
  ```python
class Portfolio:
    def __init__(self, tickers=None, start=None, end=None):
        """
        Generate a portfolio from a list of tickers.
        -------------------
        Defaults:
        Ticker: ^FTSE
        Start: 10 weeks from current date
        End: Current date
        -------------------
        Uses yahoo_finance
        """
# Setting default values to generate quick test instances
    # Use FTSE index if no ticker is provided
        if tickers==None:
            tickers = '^FTSE'
            print ('No ticker provided, FTSE was used')
        if start==None:
            start = (dt.datetime.today()-dt.timedelta(weeks=10))
            print ('Default start date: {}'.format((dt.datetime.today()-dt.timedelta(weeks=10)).strftime('%d-%m-%y')))
        if end==None:
            end = (dt.datetime.today())
            print ('Default end date: {}'.format((dt.datetime.today()).strftime('%d-%m-%y')))
        self.raw_data = yf.download(tickers, start=start, end=end)
        print('The data spans {} working days, but has {} observations.'.format(np.busday_count(start.date(),end.date()),len(self.raw_data)))
        clean_columns =[]
        self.data = {}
        for i in np.unique(self.raw_data.columns.get_level_values(0)):
                    clean_columns.append(str(i).lower().replace(" ", "_"))
        for i,x in zip(clean_columns,np.unique(self.raw_data.columns.get_level_values(0))):
            self.data[i] = self.raw_data[x]     
  ```
</details>

<details>
  <summary><b>Current version</b></summary>
  
`Current version`
```python
    # Imports
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
```
```python
tickers = ['PG', '^GSPC']
class Portfolio:
    def __init__(self, tickers=None, start=None, end=None, trading_days = 250):
        """
        Generate a portfolio from a list of tickers.
        .rawdata: {'Adj Close','Close','High','Low','Open','Volume'}
        -------------------
        tickers = []
        {start, end} = datetime
        trading_days = number of trading days in the year
        -------------------
        Defaults:
        Ticker: ^FTSE, Vodafone
        Start: 52 weeks from current date
        End: Current date
        -------------------
        Uses yahoo_finance
        """
# Setting default values to generate quick test instances
    # Use FTSE index if no ticker is provided
        if tickers==None:
            tickers = ['^FTSE','^GSPC']
            print ('No ticker provided, FTSE and S&P 500 was used')
        else: tickers = tickers
    # If no dates specified, use the range from 52 weeks ago till today
        if start==None:
            start = (dt.datetime.today()-dt.timedelta(weeks=52))
            print ('Default start date: {}'.format((dt.datetime.today()-dt.timedelta(weeks=52)).strftime('%d-%m-%y')))
        if end==None:
            end = (dt.datetime.today())
            print ('Default end date: {}'.format((dt.datetime.today()).strftime('%d-%m-%y')))
# Retieve the data from YahooFinance        
        self.raw_data = yf.download(tickers, start=start, end=end).dropna()
        self.risk_free_rate = yf.download('^TNX')['Adj Close'].iloc[-1]
# Quick indication of missing date
        print('The data spans {} working days, but has {} observations.'.format(np.busday_count(start.date(),end.date()),len(self.raw_data)))
        self.log_returns = np.log(self.raw_data['Adj Close'] / self.raw_data['Adj Close'].shift(1))
# Functions for creating portfolio returns and volatilities
    def Efficient_Frontier(self, n=1000, s=100):
        portfolio_returns = []
        portfolio_volatilities = []
        for x in range (n):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            portfolio_returns.append(np.sum(weights * self.log_returns.mean())*250)
            portfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(self.log_returns.cov() * 250, weights))))
        self.portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility':portfolio_volatilities})
        plt.figure(figsize=(10,4))
        plt.scatter(x=self.portfolios['Volatility'],y=self.portfolios['Return'],s=s)
        plt.xlabel("Volatility")
        plt.ylabel("Return")
    def equally_weighted(self):
        self.weights = np.ones(len(tickers))/len(tickers)
        self.portfolio_prices = Port_ret(self.weights, self.raw_data['Adj Close'])
        self.portfolio_return = Port_ret(self.weights, self.log_returns)
        self.portfolio_volatility = Port_vol(self.weights, self.log_returns)
    def randomly_weighted(self):
        self.weights = np.random.random(len(tickers))
        self.weights /= np.sum(self.weights)
        self.portfolio_prices = Port_ret(self.weights, self.raw_data['Adj Close'])
        self.portfolio_return = Port_ret(self.weights, self.log_returns)
        self.portfolio_volatility = Port_vol(self.weights, self.log_returns)
        
def Port_ret(weights, log_returns, trading_days=None):
    if trading_days != None:
        return (np.sum(weights * log_returns.mean()) * trading_days)
    else: return (np.sum(weights * log_returns,axis=1))
def Port_vol(weights, log_returns, trading_days=None):
    if trading_days != None:
        return (np.sqrt(np.dot(weights.T,np.dot(log_returns.cov() * trading_days, weights))))
    else: return (np.sqrt(np.dot(weights.T,np.dot(log_returns.cov(), weights))))
```
</details>

### Break down
<details>
  <summary>Break down: Log returns</summary>
      
  #### [Log returns](https://github.com/Johanlai/Main_functions/blob/main/Explanations.md#log-returns)
For calculating the daily log returns for **each** security.
```math
ln(R_i)= r_i = ln\frac{P_t}{P_{t-1}}
```
```python
log_returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
```
`Annualised`
```math
ln(R_i)= r_i = ln\frac{P_t}{P_{t-1}}
```
```python
log_returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
```
  </details>
    <details>
  <summary>Break down: Efficient frontier</summary>
  
#### Plotting the simulated efficient frontier
Generates (default = 1000) portfolios with random weights and plots their volatility vs. returns.
```python
def Efficient_Frontier(self, n=1000):
    portfolio_returns = []
    portfolio_volatilities = []
    for x in range (n):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        portfolio_returns.append(np.sum(weights * self.log_returns.mean())*250)
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(self.log_returns.cov() * 250, weights))))
    self.portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility':portfolio_volatilities})
    plt.figure(figsize=(10,6))
    plt.scatter(x=self.portfolios['Volatility'],y=self.portfolios['Return'])
    plt.xlabel("Volatility")
    plt.ylabel("Return")
```
</details>




## Models
### Binomial pricing model
Turn this into a class/function
```python
N = 15000              # number of periods or number of time steps  
payoff = "call"        # payoff 

dT = float(T) / N                             # Delta t
u = np.exp(sig * np.sqrt(dT))                 # up factor
d = 1.0 / u                                   # down factor 

V = np.zeros(N+1)                             # initialize the price vector
S_T = np.array( [(S0 * u**j * d**(N - j)) for j in range(N + 1)] )  # price S_T at time T

a = np.exp(r * dT)    # risk free compounded return
p = (a - d)/ (u - d)  # risk neutral up probability
q = 1.0 - p           # risk neutral down probability   

if payoff =="call":
    V[:] = np.maximum(S_T-K, 0.0)
else:
    V[:] = np.maximum(K-S_T, 0.0)

for i in range(N-1, -1, -1):
    V[:-1] = np.exp(-r*dT) * (p * V[1:] + q * V[:-1])    # the price vector is overwritten at each step
        
print("BS Tree Price: ", V[0])
```

### [Black–Scholes model](https://github.com/Johanlai/f_functions/blob/main/Explanations.md#blackscholes-model)
Vanilla call option - should add put functionality
```python
import numpy as np
from scipy.stats import norm
# If pulling current treasury rates data
import pandas_datareader as pdr
```
```python
class BSM:
    def __init__(self, S, k, stdev, T, r=None):
        """
        S = current stock price
        K = option strike price
        r = risk free interest rate
        stdev = sample standard deviation
        t = time until option expires
        """
        self.S = S
        self.k = k
        self.stdev = stdev
        self.T = T
        if r==None:
            self.r = float(pdr.get_data_fred('GS10').iloc[-1])
        else:
            self.r = r
    def d1(self):
        return (np.log(self.S/self.k)+(self.r+self.stdev**2/2)*self.T)/(self.stdev*np.sqrt(self.T))

    def d2(self):
        return (np.log(self.S/self.k)+(self.r-self.stdev**2/2)*self.T)/(self.stdev*np.sqrt(self.T))

    def call_price(self):
        return (self.S*norm.cdf(self.d1())) - (self.k * np.exp(-self.r * self.T) * norm.cdf(self.d2()))
```
### Monte Carlo simulations
```python
def get_data(tickers, start, end):
    data = yf.download(tickers, start, end).dropna()
    close = data['Adj Close']
    returns = close.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix
    
tickers = ['^FTSE','^GSPC']
start = (dt.datetime(2010,1,1))
end = (dt.datetime.today())

meanReturns, covMatrix = get_data(tickers, start, end)

weights = np.random.random(len(tickers))
weights /= np.sum(weights)

mc_sims = 1000
T = 10000

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T,mc_sims),fill_value=0.0)
initialPortfolio = 1000

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio
plt.plot(portfolio_sims)
plt.show()
```
