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
