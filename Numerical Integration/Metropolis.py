#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np

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

