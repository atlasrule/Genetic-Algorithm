import numpy as np


# ----- Population Initialization -----
def initializePopulation(chromosome_length, population_size):
  population = []
  individual = []
  for i in range(population_size):
    individual = np.random.choice([0, 1], size=(chromosome_length,))
    population.append(individual)

  return population
  
  
# ----- Selection -----
def tournament_selection(population, n_selections):
  winner = None
  winners = []

  for i in range(n_selections):
    chosen_index = np.random.randint(len(population))
    random_individual = population[chosen_index]

    try:
      if winner==None or fitness(random_individual) > fitness(winner):
        winner = random_individual
        winners.append(winner)
    except (ValueError, ZeroDivisionError):
      winner = random_individual
      winners.append(winner)
  
  # We need at least two parents for healty crossovers.
  # So If only one winner has picked, we randomly pick one another. 
  if len(winners) == 1:
    while True:
      chosen_index = np.random.randint(len(population))
      random_individual = population[chosen_index]
      if random_individual != winner:
        winners.append(random_individual)
        return winners  
        
  return winners


def roulette_wheel_selection(population, n_selections):
  fitnesses = [fitness(individual) for individual in population]
  probabilities = [fitness/sum(fitnesses) for fitness in fitnesses]
  chosen_indexes = np.random.choice([i for i in range(len(population))], size=(n_selections,), p=probabilities)
  if isinstance(population[0], list):
    return [population[index].tolist() for index in chosen_indexes]
  return [population[index] for index in chosen_indexes]


def selection(population, n_selections, selection_strategy='roulette'):
  if selection_strategy == 'roulette':
    return roulette_wheel_selection(population, n_selections)

  if selection_strategy == 'tournament':
    return tournament_selection(population, n_selections)
    
  return roulette_wheel_selection(population, n_selections)


# ----- Crossover -----                       
def single_point_crossover(parent_1, parent_2):
  max_point = min(len(parent_1), len(parent_2))
  random_point = np.random.randint(0, max_point)

  offspring_1 = np.concatenate( (parent_1[:random_point], parent_2[random_point:]) )
  offspring_2 = np.concatenate( (parent_2[:random_point], parent_1[random_point:]) )
  
  return offspring_1, offspring_2


def multi_point_crossover(n_points, parent_1, parent_2):
  for crossover_repetition in range(n_points):
    offspring_1 = single_point_crossover(parent_1, parent_2)
    offspring_2 = single_point_crossover(parent_1, parent_2)
  return offspring_1, offspring_2


def crossover(parent_1, parent_2, crossover_strategy='single_point'):
  if crossover_strategy == 'single_point':
    return single_point_crossover(parent_1, parent_2)

  if crossover_strategy[0].isnumeric():
    n_points = int(crossover_strategy.split('_')[0])
    multi_point_crossover(n_points, parent_1, parent_2)
 

  return single_point_crossover(parent_1, parent_2)

  
# ----- Mutation -----
# Flips one to zero or vice versa at a random point.
def bit_flip_mutation(individual):
  random_index = np.random.randint(0, len(individual))
  individual[random_index] ^= 1
  return individual


def mutation(individual, mutation_strategy='bit_flip'):
  if mutation_strategy == 'bit_flip':
    return bit_flip_mutation(individual)
  return bit_flip_mutation(individual)

  
# ----- Generation Update -----
def update_generation(population, population_size, n_selections, mutation_probability, selection_strategy, crossover_strategy, mutation_strategy):
  parents = selection(population, n_selections, selection_strategy)
  new_population = []
  n_offsprings = population_size

  offspring_1=[]
  offspring_2=[]

  while n_offsprings > 0:
    for parent_1 in parents:
      for parent_2 in parents:
        if (parent_1 != parent_2).any():
  
          offspring_1, offspring_2 = crossover(parent_1, parent_2, crossover_strategy)
          
          if np.random.choice([True, False], size=(1,), p=[mutation_probability, 1-mutation_probability]):
            offspring_1 = mutation(offspring_1, mutation_strategy)

          if np.random.choice([True, False], size=(1,), p=[mutation_probability, 1-mutation_probability]):
            offspring_2 = mutation(offspring_2, mutation_strategy)
          
          new_population.append(offspring_1)
          new_population.append(offspring_2)
          if n_offsprings <= 1:
            return new_population
          n_offsprings -= 2
          
  return new_population

  
# ----- One Function To Run Them All -----
def genetic_model(initial_population = None,
                  initial_individual = None,
                  max_simulations = 1,
                  max_iterations=80,
                  target_fitness=1,
                  chromosome_length=16,
                  population_size=400,
                  n_selections=33,
                  mutation_probability = 0.01,
                  selection_strategy='tournament',
                  crossover_strategy='single_point',
                  mutation_strategy='bit_flip'):
  
  if initial_population != None:
    population = initial_population
  else:
    population = initializePopulation(chromosome_length, population_size)

  if initial_individual != None:
    if isinstance(initial_individual, list):
      initial_individual = np.array(initial_individual)
    population.append(initial_individual)
    population_size -= 1
    
  overall_best_fitness = 0
  overall_best_individual = []
                    
  for i_simulation in range(max_simulations):

    best_fitness = 0
    best_individual = []
    
    population = update_generation(population, population_size, n_selections, mutation_probability, selection_strategy, crossover_strategy, mutation_strategy)
      
    for iteration in range(max_iterations):
      
      for individual in population:
        last_fitness = fitness(individual)
  
        if last_fitness > best_fitness:
          best_fitness = last_fitness
          best_individual = individual
    
      if best_fitness > overall_best_fitness:
        overall_best_fitness = best_fitness
        overall_best_individual = best_individual

    if overall_best_fitness >= target_fitness:
      print('Reached to target fitness.')
      break
      
    population = initializePopulation(chromosome_length, population_size)
    
    print('Simulation: {}, best fitness: {}, {}'.format(str.rjust(str(i_simulation), 3), str.rjust(str(best_fitness), 3), best_individual))
  print('Overall Best Fitness: {}, {}'.format(str.rjust(str(overall_best_fitness), 3), str.rjust(str(overall_best_individual), 3)))


def binary(binary_array):
  output = 0
  for index, digit in enumerate(reversed(binary_array)):
    output += digit * 2**(index)
  return output



# ----- Exemplary Fitness Function -----
def fitness(individual):

  if individual.any() == None:
      return 0

  # return sum(individual) / len(individual)
  nums = [binary(binary_array) for binary_array in np.array_split(individual, 4)]

  x, y, z, t = nums[0], nums[1], nums[2], nums[3]
  return np.abs(x + y + z - t)
  
#Hyper-parameters
genetic_model(max_simulations = 15,
              max_iterations=300,
              target_fitness=45,
              chromosome_length=16,
              population_size=60,
              n_selections=7,
              mutation_probability = 0.15,
              selection_strategy='roulette',
              crossover_strategy='single_point',
              mutation_strategy='bit_flip')