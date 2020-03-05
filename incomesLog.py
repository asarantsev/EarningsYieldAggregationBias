# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:12:09 2019

@author: UNR Math Stat
"""

import numpy
from numpy import linalg
from numpy import random
import pandas
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
import math

def correctLin(x, y):
    n = numpy.size(x)
    r = stats.linregress(x, y)
    s = r.slope
    i = r.intercept
    print(r)
    residuals = numpy.array([y[k] - x[k]*s - i for k in range(n)])
    stderr = math.sqrt((1/(n-2))*numpy.dot(residuals, residuals))
    qqplot(residuals, line = 'r')
    pyplot.show()
    print('Shapiro-Wilk p = ', stats.shapiro(residuals)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(residuals)[1])
    return (residuals, s, i, stderr)

df = pandas.read_excel('ModifiedEarnings.xlsx', sheet_name = 'Index')
dataR = df.values
NQTRS = 160
WINDOW = 5
NSTEPS = int(NQTRS/WINDOW)
dv = pandas.read_excel('ModifiedEarnings.xlsx', sheet_name = 'Funds')
VFINX = dv.values
TR = numpy.array([math.log(VFINX[k, 1] + VFINX[k, 2]) - math.log(VFINX[k+1, 1]) for k in range(1, 1 + NQTRS)])
TR = TR[::-1]
dg = pandas.read_excel('ModifiedEarnings.xlsx', sheet_name = 'Earnings')
earnings = dg.values
avg = earnings[:, 1]
dc = pandas.read_excel('ModifiedEarnings.xlsx', sheet_name = 'Cap')
cap = dc.values
cap = cap[:, 1]
print('')
print('')
print('Premium')
rate = dataR[::3, 3]
RF = numpy.array([math.log(1 + rate[k]/400) for k in range(NQTRS+3)])
premium = TR - RF[2:-1]
P = numpy.array([sum(premium[WINDOW*k+WINDOW+1:WINDOW*k+WINDOW*2+1]) for k in range(NSTEPS-1)])
value = [0]
for t in range(NQTRS):
    current = value[t]
    value.append(current + premium[t])
print('Wealth')
pyplot.plot(value)
pyplot.show()
print('')
print('')
lavg = numpy.array([numpy.log(item) for item in avg])
print('Averaged Positive Log Earnings')
pyplot.plot(lavg)
pyplot.show()
changes = numpy.array([lavg[k+1] - lavg[k] for k in range(NQTRS)])
print('QQ plot increments log earnings')
qqplot(changes, line = 's')
pyplot.show()
print('Shapiro-Wilk test p = ', stats.shapiro(changes)[1])
print('Jarque-Bera test p = ', stats.jarque_bera(changes)[1])
print('')
print('')
print('EY average')
EY = [math.log(avg[k]/cap[k+1]) for k in range(NQTRS)]
AvgEY = numpy.array([sum(EY[WINDOW*k:WINDOW*k+WINDOW]) for k in range(NSTEPS)])
pyplot.plot(AvgEY)
pyplot.show()
print('EY average vs premium')
pyplot.plot(AvgEY[:-1], P, 'go')
pyplot.show()
print('EY sums vs premium: linear regression')
r = correctLin(AvgEY[:-1], P)
Residuals, Slope, Intercept, Stderr = r
print('')
print('')
print('AR(1) EY sums')
ar = correctLin(AvgEY[:-1], AvgEY[1:])
ARResiduals, ARSlope, ARIntercept, ARStderr = ar
rho = numpy.corrcoef(Residuals, ARResiduals)[0, 1]
Sigma = numpy.cov(Residuals, ARResiduals)*(NSTEPS-2)
vecEstimates = [Intercept, Slope, ARIntercept, ARSlope] 
M = [[NSTEPS-1, sum(AvgEY[:-1])], [sum(AvgEY[:-1]), sum([item**2 for item in AvgEY[:-1]])]]
M = linalg.inv(M)

def iwishart(DF, Mat):
    InvMat = linalg.inv(Mat)
    output = random.multivariate_normal([0, 0], InvMat, DF+1)
    wishart = numpy.cov(output[:, 0], output[:, 1])*DF
    return linalg.inv(wishart)
    

def sim(EY, T):
    currentEY = EY
    premia = []
    simCov = iwishart(NSTEPS+1, Sigma)
    Kronecker = numpy.kron(simCov, M)
    simCoeff = random.multivariate_normal(vecEstimates, Kronecker)
    simIntercept = simCoeff[0]
    simSlope = simCoeff[1]
    simARIntercept = simCoeff[2]
    simARSlope = simCoeff[3]
    for t in range(T):
        simResiduals = random.multivariate_normal([0, 0], simCov)
        newEY = simARIntercept + currentEY * simARSlope + simResiduals[1]
        newPremium = simIntercept + simSlope * currentEY + simResiduals[0]
        premia.append(newPremium)
        currentEY = newEY
    return numpy.mean(premia)*(4/WINDOW)

NSIMS = 10000
HORIZON = int(10*(4/WINDOW))

def sims(currentEY, T):
    results = []
    for k in range(NSIMS):
        current = sim(currentEY, T)
        results.append(current)
    return (numpy.mean(results), numpy.percentile(results, 5), numpy.percentile(results, 95))

print('Future = ', sims(AvgEY[-1], HORIZON))
print('Long-term = ', sims(numpy.mean(AvgEY), HORIZON))
print('Past replay = ', sims(AvgEY[0], HORIZON))
print('Past = ', numpy.mean(P)*(4/WINDOW))
