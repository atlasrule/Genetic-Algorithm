# Genetic Algorithm
 The Classic Genetic Algorithm implementation.

## Includes
- Tournament Selection
- Roulette Selection
- Single Point Crossover
- Multi Point Crossover
- Single Bit Flip Mutation
- Stopping criteria

## Hyper-Parameters
| Parameter             | Description                                         |
| --------------------- | --------------------------------------------------- |
| n_simulations         | Number of extinctions trials                        |
| max_iterations        | Number of generations per simulation                |
| target_fitness        | Main stopping criteria of fitness                   |
| chromosome_length     | DNA storage size                                    |
| population_size       | Number of individuals per generation                |
| n_selections          | Elitism, how many individuals will have children    |
| mutation_probability  | P(mutation) for each creation                       |
| selection_strategy    | Parent determination strategy. tournament/roulette  |
| crossover_strategy    | Criteria for how offsprings will be produced        |
| mutation_strategy     | Mutation Strategy. bit-flip                         |
 
## Usage
- Replace 'fitness' function,
- Adjust parameters according to the problem's needs.