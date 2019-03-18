import torch

class LogisticRegressionModel:
    """
    Code to run an already trained logistic regression model.
    
    """    
    
    def __init__(self, w):
        """
        w is a Torch tensor storing a D-dimension vector.
        
        """
        self.w = w
        
    def predict_probs(self, X):
        """
        Given NxD evidence matrix X, this returns an N-length vector
        for which the nth element is the probability that the response
        corresponding to the nth evidence vector is equal to 1.
        
        """
        return torch.sigmoid(torch.mv(X,self.w)) 

    def classify(self, X, thres=0.5):
        """
        Given NxD evidence matrix X, this returns an N-length vector
        for which the nth element is 1 if the probability of a positive 
        response corresponding to the nth evidence vector is greater
        than the specified threshold.
        
        """
        y = self.predict_probs(X)
        for i in range(len(y)):
            if y[i] < thres:
                y[i] = 0
            else:
                y[i] = 1
        return y

    
    def evaluate(self, X, y, thres=0.5):
        """
        Given NxD evidence matrix X and the N-length expected response
        vector y, this compares the result of running classify(X, thres)
        to the expected response vector y.
        
        It returns the percentage of response values that are equivalent
        (i.e. both equal 1 or both equal 0).
        
        """
        candidates = list(self.classify(X, thres))
        expecteds = list([int(i) for i in y])
        total = 0
        correct = 0
        for (cand, exp) in zip(candidates, expecteds):
            total += 1
            if cand == exp:
                correct += 1
        return correct / total    
    
    def precision(self, X, y, thres=0.5):
        """
        Given NxD evidence matrix X and the N-length expected response
        vector y, this compares the result of running classify(X, thres)
        to the expected response vector y.
        
        It returns the fraction of the time that the classifier is 
        correct, when it makes a positive prediction (i.e. predicts
        that the response variable equals 1).
        
        """
        candidates = list(self.classify(X, thres))
        expecteds = list([int(i) for i in y])
        total = 0
        correct = 0
        for (cand, exp) in zip(candidates, expecteds):
            if cand == 1:
                total += 1
                if cand == exp:
                    correct += 1
        if total != 0:
            return correct / total
        else:
            return 1.0
        
    def recall(self, X, y, thres=0.5):
        """
        Given NxD evidence matrix X and the N-length expected response
        vector y, this compares the result of running classify(X, thres)
        to the expected response vector y.
        
        It returns the fraction of the time that the classifier is 
        correct, when it is classifying a positive instance (i.e. when
        the expected response equals 1).
        
        """
        candidates = list(self.classify(X, thres))
        expecteds = list([int(i) for i in y])
        total = 0
        correct = 0
        for (cand, exp) in zip(candidates, expecteds):
            if exp == 1:
                total += 1
                if cand == exp:
                    correct += 1
        if total != 0:
            return correct / total
        else:
            return 1.0
        
        
