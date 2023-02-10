#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
import numpy as np
import scipy.stats as spi


# In[2]:


class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
    
    def hit_rate(self):
        return(self.hits/(self.hits+self.misses))
    
    def false_alarm_rate(self):
        return(self.falseAlarms/(self.falseAlarms+self.correctRejections))
    
    def d_prime(self):
        return(spi.norm.ppf(self.hit_rate()) - spi.norm.ppf(self.false_alarm_rate()))
    
    def criterion(self):
        return(-0.5*(spi.norm.ppf(self.hit_rate()) + spi.norm.ppf(self.false_alarm_rate())))
    

