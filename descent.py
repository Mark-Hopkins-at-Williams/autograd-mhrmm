"""
Gradient descent algorithms, based on Project 1.

YOU SHOULD NOT NEED TO CHANGE THIS FILE FOR PROJECT 2.

"""

import torch

class Environment:
    """
    Interface for the unlit environment.
    
    You can:
        - determine your current (x,y) position
        - determine your current status (actively searching, 
          exceeded_step_limit, found the exit)
        - determine the gradient at a particular (x,y) position
        - step to a new (x,y) position
    
    """
        
    def __init__(self, start_pos, precision, max_steps):
        """
        Initializes the environment.
        
        start_pos is your starting position, a 2-dimensional torch.tensor
        precision is how close you need to be (in both the x and y dimension) 
          to the lowest point in order to see the exit.
        max_steps is the maximum number of steps you're willing to take
          before retreating back to the starting position
        
        """
        self.curr_pos = start_pos
        self.precision = precision
        self.steps = [start_pos]        
        self.max_steps = max_steps
    
    def gradient(self, position):
        """
        Returns the gradient at a particular position.
        
        position is a 2-dimensional torch.tensor, e.g. torch.tensor([x,y])
          where (x,y) is the current position
          
        The return value is also a 2-dimensional torch.tensor.
        
        """
        raise NotImplementedError('Cannot call .gradient() on abstract class.')

    def current_position(self):
        """
        Returns your current (x,y) position.
        
        The return value is a 2-dimensional torch.tensor, e.g.
        torch.tensor([x,y]).
        
        """
        return self.curr_pos
        
    def step_to(self, position):
        """
        Changes your current (x,y) position to the new position.
        
        position is a 2-dimensional torch.tensor, e.g. torch.tensor([x,y]).
        
        """
        self.steps.append(position)
        self.curr_pos = position
        return self.status()

    ACTIVELY_SEARCHING = 0
    EXCEEDED_STEP_LIMIT = 1
    CONVERGED = 2

    def status(self):
        """
        Returns the current status of your search.
        
        - Environment.ACTIVELY_SEARCHING means that the search is still active.
        - Environment.CONVERGED means that you have exceeded the
          maximum number of steps that you are willing to take
        - Environment.FOUND_EXIT means that you have found the lowest point
          of the environment.
        
        """
        if len(self.steps) > self.max_steps + 1:
            return Environment.EXCEEDED_STEP_LIMIT
        elif (len(self.steps) >= 2 and 
              torch.max(torch.abs(self.steps[-1] - self.steps[-2])) < self.precision):
            return Environment.CONVERGED
        else:
            return Environment.ACTIVELY_SEARCHING
    

def grad_descent(step_fn, env):
    """
    A general-purpose gradient descent algorithm.
    
    step_fn is a function that takes a position (x,y) as input (expressed as
    a 2-dimensional torch.tensor), and returns the relative step to take
    (also expressed as a 2-dimensional torch.tensor).
    
    env is the environment.
    
    The return value should be a list of the positions (including the starting
    positions) visited during the gradient descent. 
    
    """
    steps = [env.current_position()]
    while env.status() == Environment.ACTIVELY_SEARCHING:
        step = step_fn(env.current_position())
        #next_pos = torch.tensor(env.current_position() + step, requires_grad=True)
        next_pos = (env.current_position() + step).clone().detach().requires_grad_(True)
        env.step_to(next_pos)
        steps.append(next_pos)
    return steps


class AdagradStepFunction:
    """
    Computes the next step for adagrad.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that adagrad would take (also expressed as a
    2-dimensional torch.tensor).
        
    """
    def __init__(self, loss_gradient, init_learning_rate, delta = 0.0000001):
        self.sum_squared_gradients = 0.0
        self.delta = delta
        self.loss_gradient = loss_gradient
        self.init_learning_rate = init_learning_rate
        
    def __call__(self, pos):
        gradient = self.loss_gradient(pos)
        squared_gradient = gradient * gradient
        self.sum_squared_gradients += squared_gradient
        learning_rate = (self.init_learning_rate / 
                         (self.delta + 
                          torch.sqrt(self.sum_squared_gradients)))
        result = (-learning_rate * gradient)
        return result.double()

class VanillaStepFunction:
 
    def __init__(self, loss_gradient, learning_rate):
        
        self.loss_gradient = loss_gradient
        self.learning_rate = learning_rate
        
    def __call__(self, pos):
        gradient = self.loss_gradient(pos)
        result = (-self.learning_rate * gradient)
        return result.double()

def adagrad(rate, env):
    return grad_descent(AdagradStepFunction(env.gradient, rate), env)

def vanilla_grad_descent(rate, env):
    return grad_descent(VanillaStepFunction(env.gradient, rate), env)

