#### Log returns
Logging the returns aids to normalise the data for analysis
```math
ln(R_i)= r_i = ln\frac{P_t}{P_{t-1}}
```
- Log returns are (assumed to be) normally distributed
- They are additive across time


# Black–Scholes model
```math
C(S,t) = N(d_1)S - N(d_2)Ke^{-r(T-t)}\newline
```
where<br>
$d_1 = \frac{1}{s\sqrt{(T-t)}}[ln\frac{S}{K}+(r+\frac{s^2}{2})(T-t)]$ is the expected return if the option is exercised<br>
$d_2 = \frac{1}{s\sqrt{(T-t)}}[ln\frac{S}{K}+(r-\frac{s^2}{2})(T-t)] = d_1 - s\sqrt{T-t}$ is the cost of exercising the option
<br><br>
S = current stock price<br>
K = option strike price<br>
t = time until option expires<br>
r = risk free interest rate<br>
s = sample standard deviation<br>
N = standard normal distribution<br>
e = exponential term<br>
C = call premium

In the merton model, the probability of default is simply the probability function of the normal minus the distance to default. $PD = \mathcal{N}(-d_2)$
