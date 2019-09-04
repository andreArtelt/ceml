# -*- coding: utf-8 -*-
import numpy as np
from random import randint, choice
from .optimizer import Optimizer


class EvolutionaryOptimizer(Optimizer):
    """
    Evolutionary/Genetic optimization algorithm.

    Note
    ----
    This genetic algorithm is a gradient-free optimization algorithm.

    This implementation encodes an individual as a `numpy.array` - if you want to use a different representation, you have to derive a new class from this class and reimplement all relevant methods.

    Parameters
    ----------
    population_size : `int`
        The size of the population

        The default is 100
    select_by_fitness : `float`
        The fraction of individuals that is selected according to their fitness.

        The default is 0.5
    mutation_prob : `float`
        The proability that an offspring is mutated.

        The default is 0.1
    mutation_scaling : `float`
        Standard deviation of the normal distribution for mutating  features.

        The default is 4.0
    """
    def __init__(self, population_size=100, select_by_fitness=0.5, mutation_prob=0.1, mutation_scaling=4.):
        self.population = []
        self.population_size = population_size
        self.select_by_fitness = select_by_fitness
        self.mutation_prob = mutation_prob
        self.mutation_scaling = mutation_scaling

        self.f = None
        self.x0 = None
        self.tol = None
        self.max_iter = None

        super(EvolutionaryOptimizer, self).__init__()
    
    def init(self, f, x0, tol=None, max_iter=None):
        """
        Initializes all remaining parameters.

        Parameters
        ----------
        f : `callable`
            The objective that is minimized.
        x0 : `numpy.array`
            The initial value of the unknown variable.
        tol : `float`, optional
            Tolerance for termination.

            `tol=None` is equivalent to `tol=0`.

            The default is 0.
        max_iter : `int`, optional
            Maximum number of iterations.

            If `max_iter` is None, the default value of the particular optimization algorithm is used.

            Default is None.
        """
        self.f = f
        self.x0 = x0
        self.tol = tol if tol is not None else 0.
        self.max_iter = max_iter if max_iter is not None else 100

    def is_grad_based(self):
        return False

    def __call__(self):
        return self.optimize()

    # *******************************************
    # * Below: Methods of the genetic algorithm *
    # *******************************************

    def crossover(self, x0, x1):
        """
        Produces an offspring from the individuals `x0` and `x1`.

        Note
        ----
        This method implements **single-point crossover**. If you want to use a different crossover strategy, you have to derive a new class from this one and reimplement the method `crossover`

        Parameters
        ----------
        x0 : `numpy.array`
            The representation of first individual.
        x1 : `numpy.array`
            The representation of second individual.
        
        Returns
        -------
        `numpy.array`
            The representation of offspring created from `x0` and `x1`.
        """
        # Choose a random crossover point
        p = randint(0, x0.shape[0])

        # Compute offspring
        return np.concatenate((x0[:p], x1[p:]), axis=0)

    def mutate(self, x):
        """
        Mutates a given individual `x`.

        Parameters
        ----------
        x : `numpy.array`
            The representation of the individual.
        
        Returns
        -------
        `numpy.array`
            The representation of the mutated individual `x`.
        """
        for i in range(x.shape[0]):
            if np.random.uniform() <= self.mutation_prob:
                x[i] += np.random.normal(scale=self.mutation_scaling)

        return x
    
    def validate(self, x):
        """
        Validates a given individual `x`.

        This methods checks whether a given individual is valid (in the sense that the feature characteristics are valid) and if not it makes it valid by changing some of its features.

        Note
        ----
        This implementation is equivalent to the identity function. The input is returned without any changes - we do not restrict the input space!
        If you want to make some restrictions on the input space, you have to derive a new class from this one and reimplement the method `validate`.

        Parameters
        ----------
        x : `numpy.array`
            The representation of the individual `x`.

        Returns
        -------
        `numpy.array`
            The representation of the validated individual.
        """
        return x
    
    def compute_fitness(self, x):
        """
        Computes the fitness of a given individual `x`.

        Parameters
        ----------
        x : `numpy.array`
            The representation of the individual.
        """
        return -1. * self.f(x)  # Note: We can not use the objective function for computing fitness score because a genetic algorithm maximizes the fitness - but we want to minimize the function! However, minimizing a function is equivalent to maximizing the negative function.

    def select_candidates(self, fitness):
        """
        Selects a the most fittest individuals from the current population for producing offsprings.

        Parameters
        ----------
        fitness : `list(float)`
            Fitness of the individuals.

        Returns
        -------
        `list(numpy.array)`
            The selected individuals.
        """
        # Select a proportion of the fittest individuals
        fitest = np.argsort(fitness)[::-1]
        n = int(self.population_size * self.select_by_fitness)
        
        return [self.population[i] for i in fitest[:n]]

    def optimize(self):
        # Initialize population
        self.population = [self.mutate(np.array(self.x0)) for _ in range(self.population_size)]
        
        # Keep track of the best solution
        fitness = [self.compute_fitness(x) for x in self.population]
        i = np.argsort(fitness)[-1]
        best_score = fitness[i]
        best_sample = self.population[i]

        # Run evolution
        for _ in range(self.max_iter):
            # Select parents
            self.population = self.select_candidates(fitness)

            # Produce offsprings
            offsprings = []
            for _ in range(len(self.population) - self.population_size):
                x0 = choice(self.population)
                x1 = choice(self.population)
                
                offsprings.append(self.mutate(self.crossover(x0, x1)))
            self.population += offsprings

            # Keep track of the best solution
            fitness = [self.compute_fitness(x) for x in self.population]
            i = np.argsort(fitness)[-1]
            if fitness[i] > best_score:
                best_score = fitness[i]
                best_sample = self.population[i]
   
        return best_sample
