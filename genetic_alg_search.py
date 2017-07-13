# Genetic Algorithm Search for Vs (and future Q)
# VERSION: 0.1
# AUTHOR(S): JAMES HOLT - UNIVERSITY OF LIVERPOOL
#
# EMAIL: j.holt@liverpool.ac.uk
#
#
# ---------------------------------modules--------------------------------------#
import random
import copy
import datetime
import numpy as np
import sim1D as sd
from utils import cor_v_space


def display_progress(genes, fitness, grade, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{0}\t{1}\t{2}\t{3}".format(genes, fitness, grade, str(timeDiff)))


def mutate(individual, sampleSpace, learning_rate=1):
    child = copy.deepcopy(individual)
    indexes = [random.randrange(0, len(individual) - 1) for _ in range(learning_rate)]
    for index in indexes:
        newGene, alternate = random.sample(sampleSpace[:, index].tolist(), 2)
        # newGene, alternate = random.sample(np.linspace(100, 1500, 141).tolist(), 2)
        if newGene == individual[index]:
            child[index] = alternate
        else:
            child[index] = newGene

    return child


def fitness(individual, target):
    # First modify the site model to be prepared for initiation of forward model.
    VLER10.modify_site_model(individual, q_model=True, rect_space=True)
    # In one step calculate the forward model, evaluate the RMS error and return.
    return 1 / np.sqrt(np.mean((VLER10.linear_forward_model_1d(elastic=False) - target) ** 2))


def pop_grade(population, target):
    # Evaluate the fitness for each member of the population
    graded = [(fitness(individual, target), individual) for individual in population]
    # Calculate the mean 'health' (success) of the population
    grade = np.mean([fit[0] for fit in graded])
    return grade, graded


def evolve(population, target, sampleSpace, retain=0.2, random_select=0.05, mutate_chance=0.01):
    # TODO: Finish the evolution function, evolution should include ordering the population from best[0]-worst[-1]...
    # ... Then, we need to choose the first 20% (best) of models as parents for the next generation and randomly choose
    # ... some poorer performers to include in the next generation population as parents.
    # We then must mutate some members to promote diversity (function already designed). Not each individual must be
    # ... mutated so must randomise this process.
    # We then cross-over the parents to create children to fill remaining spaces in the population. Children share
    # 'DNA' from parents in the population. 'DNA' will be mixed as evenly as possible (50/50).
    grade, graded = pop_grade(population, target)
    # this will organise the tuples from highest to lowest fitness - uses the first element of tuple to sort list
    graded = sorted(graded, key=lambda x: x[0], reverse=True)
    bestFitness, bestModel = graded[0]
    graded = [x[1] for x in graded]
    # parents are chosen as best x% of the previous generation
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    # now randomly select some other members of the previous generation for the parent generation.
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)

    # now parents are chosen, we mutate them at random

    for i, individual in enumerate(parents):
        if mutate_chance > random.random():
            parents[i] = mutate(individual, sampleSpace[0])

    parents_len = len(parents)
    remaining_pop_space = len(population) - parents_len

    children = []

    while len(children) < remaining_pop_space:
        male = random.randrange(0, parents_len)
        female = random.randrange(0, parents_len)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)  # might not exactly be half (in fact very unlikely)
            child = np.hstack((male[:half], female[half:]))
            children.append(child)
    parents.extend(children)
    return parents, grade, bestFitness, bestModel


def optimise_model(population, target, sampleSpace, retain=0.2, random_select=0.05, mutate_chance=0.5,
                   trial_cap=100000):
    trials = 0
    bestFitness = 0
    bestParent = 0
    startTime = datetime.datetime.now()

    while True:
        trials += 1
        if (bestFitness >= 10000) or trials == trial_cap:
            break
        population, grade, bestFitness, bestParent = evolve(population, target, sampleSpace, retain, random_select,
                                                            mutate_chance)
        display_progress(bestParent, bestFitness, grade, startTime)  # I want to see only positive progress
    # We escape the while loop

    return bestParent, bestFitness


def gen_initial_population(sampleSpace, pop_size=100):
    """
    Individual members (stochastic Vs models) are chosen at random from the sample space.
    They are returned as a list to build a population for validation, cross-over and mutation.
    :param sampleSpace: A matrix containing stochastic Vs models where cols are a single velocity band and
            rows are complete Vs models.
    :param pop_size: The size of the population to be generated.
    :return:
    """
    i = [random.randrange(0, len(sampleSpace)) for _ in range(pop_size)]
    return [sampleSpace[0][i[x]] for x in range(pop_size - 1)] + [sampleSpace[1]]


def main():
    VLER10 = sd.Sim1D('VLER10', '/data/share/Japan/SiteInfo/S_B/VLER10_blind/')

    mods = 100000
    sample_space = cor_v_space(v_mod=VLER10.vel_profile['vs'],
                               hl_profile=VLER10.vel_profile['thickness'], count=mods, lower_v=100, upper_v=1500,
                               cor_co=0.75, scale=1, repeat_layers=True, repeat_chance=0.5)
    VLER10.reset_site_model()
    VLER10.modify_site_model(sample_space[1] + 100, q_model=True, rect_space=True)
    target = VLER10.linear_forward_model_1d(elastic=False)
    # VLER10.modify_site_model(sample_space[1] + 100, q_model=True, rect_space=True)
    # guess = VLER10.linear_forward_model_1d(elastic=False)
    population = gen_initial_population(sample_space, pop_size=50)
    bestParent, bestFitness = optimise_model(population, target, sample_space)
    return bestParent, bestFitness


if __name__ == '__main__':
    main()
