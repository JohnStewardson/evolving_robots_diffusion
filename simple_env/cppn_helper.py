import neat
import numpy as np

class Population(neat.Population):

    def run(self, fitness_function, constraint_function=None, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided constraint function.
            # If some genomes violate the constraint, generate new genomes and replace them, until all genomes satisfy the constraint.
            if constraint_function is not None:
                genomes = list(self.population.items())
                validity = constraint_function(genomes, self.config, self.generation)
                valid_idx = np.where(validity)[0]
                valid_genomes = np.array(genomes)[valid_idx]
                while len(valid_genomes) < self.config.pop_size:
                    new_population = self.reproduction.create_new(self.config.genome_type,
                                                                    self.config.genome_config,
                                                                    self.config.pop_size - len(valid_genomes))
                    new_genomes = list(new_population.items())
                    validity = constraint_function(new_genomes, self.config, self.generation)
                    valid_idx = np.where(validity)[0]
                    valid_genomes = np.vstack([valid_genomes, np.array(new_genomes)[valid_idx]])

                self.population = dict(valid_genomes)
                self.species.speciate(self.config, self.population, self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config, self.generation)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome


"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
import multiprocessing as mp

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

# class Pool(mp.pool.Pool):
#     Process = NoDaemonProcess

class NonDaemonPool(mp.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


class ParallelEvaluator(object):
    def __init__(self, num_workers, fitness_function, constraint_function=None, timeout=None):
        """
        fitness_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        constraint_function should take one argument, a tuple of
        (genome object, config object), and return
        a single bool (the genome's validity).
        """
        self.num_workers = num_workers
        self.fitness_function = fitness_function
        self.constraint_function = constraint_function
        self.timeout = timeout
        self.pool = NonDaemonPool(num_workers)

    def __del__(self):
        self.pool.terminate() # should this be terminate?
        self.pool.join()

    def evaluate_fitness(self, genomes, config, generation):
        jobs = []
        for i, (_, genome) in enumerate(genomes):
            jobs.append(self.pool.apply_async(self.fitness_function, (genome, config, i, generation)))

        # assign the fitness back to each genome
        for job, (_, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)

    def evaluate_constraint(self, genomes, config, generation):
        validity_all = []
        for i, (_, genome) in enumerate(genomes):
            validity = self.constraint_function(genome, config, i, generation)
            validity_all.append(validity)
        return validity_all
