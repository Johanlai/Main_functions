### Black–Scholes model
Vanilla call option
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
