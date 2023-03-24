#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

