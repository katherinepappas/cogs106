#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

class SignalDetection:
    
    #constructor method to initialize SignalDetection object

    def __init__(self, hits, misses, false_alarms, correct_rejections):
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections
    
    #defines hit rate
    
    def hit_rate(self):
        return(self.hits/(self.hits+self.misses))
    
    #defines false alarm rate
    
    def false_alarm_rate(self):
        return (self.false_alarms/(self.false_alarms + self.correct_rejections))
    
    #defines d'
    
    def dprime(self):
        return (norm.ppf(self.hit_rate()) - norm.ppf(self.false_alarm_rate()))
    
    #defines criterion
    
    def criterion(self):
        return((-0.5)*(norm.ppf(self.hit_rate()) + norm.ppf(self.false_alarm_rate())))
    
    #overloads + operator for SignalDetection class
    
    def __add__(self, other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.false_alarms + other.false_alarms, self.correct_rejections + other.correct_rejections)

    #overloads * operator for SignalDetection class
    
    def __mul__(self, scalar):
        return SignalDetection(self.hits * scalar, self.misses * scalar, self.false_alarms * scalar, self.correct_rejections * scalar)

    #static factory method that creates SignalDetection objects with simulated data
    
    @staticmethod
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        sdtList = list()
        for i in range(len(criteriaList)):
            Criterion = criteriaList[i] + (dprime/2)
            hitRate = 1-norm.cdf(Criterion-dprime)
            false_alarm_rate = 1-norm.cdf(Criterion)
            hits, falseAlarms = np.random.binomial(n=[signalCount,noiseCount], p=[hitRate, false_alarm_rate])
            misses, correctRejections = signalCount-hits, noiseCount-falseAlarms
            sdtList.append(SignalDetection(hits,misses,falseAlarms,correctRejections))
        return sdtList
    
    #static method that takes SignalDetection objects and plots ROC Curve
    
    @staticmethod 
    def plot_roc(sdtList):
        for sdt in sdtList:
            hitRate = sdt.hits / (sdt.hits + sdt.misses)
            falseAlarmRate = sdt.false_alarms / (sdt.false_alarms + sdt.correct_rejections)
            plt.plot(sdt.false_alarm_rate(), sdt.hit_rate(), 'o', color='black')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.plot([0,1,100], [0,1,100], '--', color='black')
        
    #instance method that calculates negative log-likelihood of a SignalDetection object
    
    def nLogLikelihood(self, hitRate, falseAlarmRate):
        return -self.hits*np.log(hitRate)-self.misses*np.log(1-hitRate)-self.false_alarms*np.log(falseAlarmRate)-self.correct_rejections*np.log(1-falseAlarmRate)
      
    #static method that computes one-parameter ROC curve function
    
    @staticmethod
    def rocCurve(falseAlarmRate, a):
        return norm.cdf(a+norm.ppf(falseAlarmRate))
        
    #static method that fits one-paramter function to observed [hit rate, false alarm rate] pairs
    
    @staticmethod
    def fit_roc(sdtList):
        def objective(a):
            return SignalDetection.rocLoss(a, sdtList)
        fit = minimize_scalar(objective, bounds=(0,1), method='nelder-mead', args=(sdtList))
        curve = list()
        for i in range(0,100,1):
            curve.append((SignalDetection.rocCurve(i, float(fit.x))))
        plt.plot([0,1,100], [0,1,100], curve, '-', color='red')
        aHat = fit.x
        return float(aHat)
    
    #static method that evaluates loss function
    
    @staticmethod
    def rocLoss(a, sdtList):
        L = list()
        for i in range(len(sdtList)):
            predHitRate = (SignalDetection.rocCurve(sdtList[i].false_alarm_rate(), a))
            L.append(sdtList[i].nLogLikelihood(predHitRate, sdtList[i].false_alarm_rate()))
        return sum(L)
    
class Metropolis:
    
    #constructor method to initialize Metropolis object 
    
    def __init__(self, logTarget, initialState, stepSize=1.0):
        self.logTarget = logTarget
        self.state = initialState
        self.stepSize = stepSize
        self.acceptCount = 0
        self.totalCount = 0
        self.samples = [self.state]
        
    #private method to check whether to accept or reject proposed value 
        
    def _accept(self, proposal):
        logAcceptRatio = self.logTarget(proposal) - self.logTarget(self.state)
        if np.log(np.random.rand()) < logAcceptRatio:
            self.state = proposal
            self.acceptCount += 1
            self.samples.append(self.state)
            return True
        else:
            self.samples.append(self.state)
            return False
    
    #performs adaptation phase of Metropolis algorithm
    
    def adapt(self, blockLengths):
        for blockLength in blockLengths:
            for i in range(blockLength):
                proposal = np.random.normal(loc=self.state, scale=self.stepSize)
                self.totalCount += 1
                self._accept(proposal)
            acceptanceRate = self.acceptCount / self.totalCount
            if acceptanceRate < 0.1:
                self.stepSize /= 2
                self.acceptCount = 0
                self.totalCount = 0
            elif acceptanceRate > 0.5:
                self.stepSize *= 2
                self.acceptCount = 0
                self.totalCount = 0
        return self
    
    #uses Metropolis algorithm to generate n samples from the target distribution
    
    def sample(self, nSamples):
        for i in range(nSamples):
            proposal = np.random.normal(loc=self.state, scale=self.stepSize)
            self.totalCount += 1
            self._accept(proposal)
        return self
    
    #returns mean and 95% credible interval of generated samples
    
    def summary(self):
        mean = np.mean(self.samples)
        c025 = np.percentile(self.samples, 2.5)
        c975 = np.percentile(self.samples, 97.5)
        return {'mean': mean, 'c025': c025, 'c975': c975}


# In[46]:


import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def fit_roc_bayesian(sdtList):

    # Define the log-likelihood function to optimize
    def loglik(a):
        return -SignalDetection.rocLoss(a, sdtList) + scipy.stats.norm.logpdf(a, loc = 0, scale = 10)

    # Create a Metropolis sampler object and adapt it to the target distribution
    sampler = Metropolis(logTarget = loglik, initialState = 0)
    sampler = sampler.adapt(blockLengths = [2000]*3)

    # Sample from the target distribution
    sampler = sampler.sample(nSamples = 4000)

    # Compute the summary statistics of the samples
    result  = sampler.summary()

    # Print the estimated value of the parameter a and its credible interval
    print(f"Estimated a: {result['mean']} ({result['c025']}, {result['c975']})")

    # Create a mosaic plot with four subplots
    fig, axes = plt.subplot_mosaic(
        [["ROC curve", "ROC curve", "traceplot"],
         ["ROC curve", "ROC curve", "histogram"]],
        constrained_layout = True
    )

    # Plot the ROC curve of the SDT data
    plt.sca(axes["ROC curve"])
    SignalDetection.plot_roc(sdtList = sdtList)

    # Compute the ROC curve for the estimated value of a and plot it
    xaxis = np.arange(start = 0.00,
                      stop  = 1.00,
                      step  = 0.01)

    plt.plot(xaxis, SignalDetection.rocCurve(xaxis, result['mean']), 'r-')

    # Shade the area between the lower and upper bounds of the credible interval
    plt.fill_between(x  = xaxis,
                     y1 = SignalDetection.rocCurve(xaxis, result['c025']),
                     y2 = SignalDetection.rocCurve(xaxis, result['c975']),
                     facecolor = 'r',
                     alpha     = 0.1)

    # Plot the trace of the sampler
    plt.sca(axes["traceplot"])
    plt.plot(sampler.samples)
    plt.xlabel('iteration')
    plt.ylabel('a')
    plt.title('Trace plot')

    # Plot the histogram of the samples
    plt.sca(axes["histogram"])
    plt.hist(sampler.samples,
             bins    = 51,
             density = True)
    plt.xlabel('a')
    plt.ylabel('density')
    plt.title('Histogram')

    # Show the plot
    plt.show()

# Define the number of SDT trials and generate a simulated dataset
sdtList = SignalDetection.simulate(dprime       = 1,
                                   criteriaList = [-1, 0, 1],
                                   signalCount  = 40,
                                   noiseCount   = 40)

# Fit the ROC curve to the simulated dataset
fit_roc_bayesian(sdtList)

