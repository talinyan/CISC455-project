from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Location:
    """
    Represents one delivery location or the depot.
    """
    name: str
    x: float
    y: float


@dataclass
class RouteIndividual:
    """
    Represents one candidate postal delivery route.
    chromosome:
        A permutation of delivery location indices.
        Example: [2, 0, 3, 1]
    total_distance:
        Total route distance including depot -> first stop and last stop -> depot
    fitness:
        Higher is better. Defined as 1 / total_distance.
    """
    chromosome: List[int]
    total_distance: float = math.inf
    fitness: float = 0.0


class PostalDeliveryEA:
    """
    Evolutionary Algorithm for optimizing a postal truck delivery route.
    The truck starts at the depot, visits all delivery locations exactly once,
    and returns to the depot.
    """

    def __init__(
        self,
        depot: Location,
        delivery_locations: List[Location],
        population_size: int = 100,
        generations: int = 300,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        elite_count: int = 2,
        random_seed: int | None = 42,
    ) -> None:
        if not delivery_locations:
            raise ValueError("delivery_locations must contain at least one location.")
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if generations < 1:
            raise ValueError("generations must be at least 1.")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if tournament_size < 2:
            raise ValueError("tournament_size must be at least 2.")
        if elite_count < 0 or elite_count >= population_size:
            raise ValueError("elite_count must be between 0 and population_size - 1.")

        self.depot = depot
        self.delivery_locations = delivery_locations
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count

        if random_seed is not None:
            random.seed(random_seed)

        self.distance_matrix = self._build_distance_matrix()

    def _euclidean_distance(self, first: Location, second: Location) -> float:
        """
        Computes Euclidean distance between two locations.
        """
        return math.sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2)

    def _build_distance_matrix(self) -> List[List[float]]:
        """
        Builds a distance matrix for delivery locations only.
        The depot distances are handled separately.
        """
        number_of_locations = len(self.delivery_locations)
        matrix = [[0.0 for _ in range(number_of_locations)] for _ in range(number_of_locations)]

        for i in range(number_of_locations):
            for j in range(number_of_locations):
                if i != j:
                    matrix[i][j] = self._euclidean_distance(
                        self.delivery_locations[i],
                        self.delivery_locations[j],
                    )
        return matrix

    def _distance_from_depot(self, location_index: int) -> float:
        """
        Returns distance from the depot to a delivery location.
        """
        return self._euclidean_distance(self.depot, self.delivery_locations[location_index])

    def create_random_individual(self) -> RouteIndividual:
        """
        Creates one random valid delivery route.
        """
        chromosome = list(range(len(self.delivery_locations)))
        random.shuffle(chromosome)
        individual = RouteIndividual(chromosome=chromosome)
        self.evaluate_individual(individual)
        return individual

    def initialize_population(self) -> List[RouteIndividual]:
        """
        Creates the initial population of random routes.
        """
        return [self.create_random_individual() for _ in range(self.population_size)]

    def calculate_total_route_distance(self, chromosome: List[int]) -> float:
        """
        Calculates total distance of a route:
        depot -> first stop -> ... -> last stop -> depot
        """
        if not chromosome:
            return 0.0

        total_distance = 0.0

        # Distance from depot to first delivery stop
        total_distance += self._distance_from_depot(chromosome[0])

        # Distances between consecutive delivery stops
        for current_position in range(len(chromosome) - 1):
            current_stop_index = chromosome[current_position]
            next_stop_index = chromosome[current_position + 1]
            total_distance += self.distance_matrix[current_stop_index][next_stop_index]

        # Distance from last delivery stop back to depot
        total_distance += self._distance_from_depot(chromosome[-1])

        return total_distance

    def evaluate_individual(self, individual: RouteIndividual) -> None:
        """
        Updates total distance and fitness for a route individual.
        """
        individual.total_distance = self.calculate_total_route_distance(individual.chromosome)

        if individual.total_distance == 0:
            individual.fitness = float("inf")
        else:
            individual.fitness = 1.0 / individual.total_distance

    def evaluate_population(self, population: List[RouteIndividual]) -> None:
        """
        Evaluates all individuals in the population.
        """
        for individual in population:
            self.evaluate_individual(individual)

    def tournament_selection(self, population: List[RouteIndividual]) -> RouteIndividual:
        """
        Selects one parent using tournament selection.
        """
        tournament_group = random.sample(population, self.tournament_size)
        return max(tournament_group, key=lambda individual: individual.fitness)

    def ordered_crossover(
        self,
        first_parent: RouteIndividual,
        second_parent: RouteIndividual,
    ) -> Tuple[RouteIndividual, RouteIndividual]:
        """
        Performs ordered crossover (OX) for permutation-based routes.
        Returns two children.
        """
        chromosome_length = len(first_parent.chromosome)

        if chromosome_length < 2:
            return (
                RouteIndividual(chromosome=first_parent.chromosome[:]),
                RouteIndividual(chromosome=second_parent.chromosome[:]),
            )

        start_index, end_index = sorted(random.sample(range(chromosome_length), 2))

        first_child_chromosome = [-1] * chromosome_length
        second_child_chromosome = [-1] * chromosome_length

        # Copy segment from parents
        first_child_chromosome[start_index:end_index + 1] = first_parent.chromosome[start_index:end_index + 1]
        second_child_chromosome[start_index:end_index + 1] = second_parent.chromosome[start_index:end_index + 1]

        # Fill remaining positions from the opposite parent in order
        self._fill_ordered_crossover_child(
            child_chromosome=first_child_chromosome,
            donor_chromosome=second_parent.chromosome,
            start_index=start_index,
            end_index=end_index,
        )
        self._fill_ordered_crossover_child(
            child_chromosome=second_child_chromosome,
            donor_chromosome=first_parent.chromosome,
            start_index=start_index,
            end_index=end_index,
        )

        first_child = RouteIndividual(chromosome=first_child_chromosome)
        second_child = RouteIndividual(chromosome=second_child_chromosome)

        return first_child, second_child

    def _fill_ordered_crossover_child(
        self,
        child_chromosome: List[int],
        donor_chromosome: List[int],
        start_index: int,
        end_index: int,
    ) -> None:
        """
        Helper method for ordered crossover.
        Fills empty slots in child_chromosome using donor order.
        """
        chromosome_length = len(child_chromosome)
        donor_values = [gene for gene in donor_chromosome if gene not in child_chromosome]

        insertion_index = (end_index + 1) % chromosome_length
        donor_pointer = 0

        while donor_pointer < len(donor_values):
            if child_chromosome[insertion_index] == -1:
                child_chromosome[insertion_index] = donor_values[donor_pointer]
                donor_pointer += 1
            insertion_index = (insertion_index + 1) % chromosome_length

    def swap_mutation(self, individual: RouteIndividual) -> None:
        """
        Performs swap mutation by exchanging two delivery stops.
        """
        chromosome_length = len(individual.chromosome)
        if chromosome_length < 2:
            return

        first_index, second_index = random.sample(range(chromosome_length), 2)
        individual.chromosome[first_index], individual.chromosome[second_index] = (
            individual.chromosome[second_index],
            individual.chromosome[first_index],
        )

    def create_next_generation(self, current_population: List[RouteIndividual]) -> List[RouteIndividual]:
        """
        Creates the next generation using elitism, selection, crossover, and mutation.
        """
        sorted_population = sorted(
            current_population,
            key=lambda individual: individual.fitness,
            reverse=True,
        )

        next_generation: List[RouteIndividual] = []

        # Elitism: directly preserve the best individuals
        for elite_individual in sorted_population[:self.elite_count]:
            elite_copy = RouteIndividual(chromosome=elite_individual.chromosome[:])
            elite_copy.total_distance = elite_individual.total_distance
            elite_copy.fitness = elite_individual.fitness
            next_generation.append(elite_copy)

        # Fill the rest of the next generation
        while len(next_generation) < self.population_size:
            first_parent = self.tournament_selection(current_population)
            second_parent = self.tournament_selection(current_population)

            if random.random() < self.crossover_rate:
                first_child, second_child = self.ordered_crossover(first_parent, second_parent)
            else:
                first_child = RouteIndividual(chromosome=first_parent.chromosome[:])
                second_child = RouteIndividual(chromosome=second_parent.chromosome[:])

            if random.random() < self.mutation_rate:
                self.swap_mutation(first_child)

            if random.random() < self.mutation_rate:
                self.swap_mutation(second_child)

            self.evaluate_individual(first_child)
            self.evaluate_individual(second_child)

            next_generation.append(first_child)
            if len(next_generation) < self.population_size:
                next_generation.append(second_child)

        return next_generation

    def get_best_individual(self, population: List[RouteIndividual]) -> RouteIndividual:
        """
        Returns the best individual from the population.
        """
        return max(population, key=lambda individual: individual.fitness)

    def run(self) -> Tuple[RouteIndividual, List[float]]:
        """
        Runs the evolutionary algorithm.
        Returns:
            best_individual: the best delivery route found
            best_distance_history: best route distance in each generation
        """
        population = self.initialize_population()
        best_distance_history: List[float] = []

        overall_best = self.get_best_individual(population)

        for generation_number in range(self.generations):
            current_best = self.get_best_individual(population)

            if current_best.fitness > overall_best.fitness:
                overall_best = RouteIndividual(chromosome=current_best.chromosome[:])
                overall_best.total_distance = current_best.total_distance
                overall_best.fitness = current_best.fitness

            best_distance_history.append(current_best.total_distance)

            population = self.create_next_generation(population)

            if (generation_number + 1) % 50 == 0 or generation_number == 0:
                print(
                    f"Generation {generation_number + 1:3d} | "
                    f"Best route distance: {current_best.total_distance:.2f}"
                )

        final_best = self.get_best_individual(population)
        if final_best.fitness > overall_best.fitness:
            overall_best = final_best

        return overall_best, best_distance_history

    def format_route_for_display(self, chromosome: List[int]) -> str:
        """
        Converts a chromosome into a readable postal delivery route string.
        """
        route_names = ["Depot"]
        route_names.extend(self.delivery_locations[index].name for index in chromosome)
        route_names.append("Depot")
        return " -> ".join(route_names)


def build_sample_postal_delivery_data() -> Tuple[Location, List[Location]]:
    """
    Creates sample postal delivery locations.
    You can replace these with real coordinates or your own dataset.
    """
    depot = Location(name="Postal Depot", x=50, y=50)

    delivery_locations = [
        Location(name="House A", x=20, y=65),
        Location(name="House B", x=80, y=20),
        Location(name="Apartment C", x=15, y=30),
        Location(name="Office D", x=60, y=85),
        Location(name="Shop E", x=95, y=60),
        Location(name="School F", x=40, y=10),
        Location(name="Clinic G", x=10, y=85),
        Location(name="Library H", x=75, y=75),
    ]

    return depot, delivery_locations


def main() -> None:
    depot, delivery_locations = build_sample_postal_delivery_data()

    postal_delivery_solver = PostalDeliveryEA(
        depot=depot,
        delivery_locations=delivery_locations,
        population_size=100,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.2,
        tournament_size=3,
        elite_count=2,
        random_seed=42,
    )

    best_route, best_distance_history = postal_delivery_solver.run()

    print("\nBest postal delivery route found:")
    print(postal_delivery_solver.format_route_for_display(best_route.chromosome))
    print(f"\nTotal route distance: {best_route.total_distance:.2f}")

    print("\nBest route distance progression (first 10 generations):")
    print([round(distance, 2) for distance in best_distance_history[:10]])

    print("\nBest route distance progression (last 10 generations):")
    print([round(distance, 2) for distance in best_distance_history[-10:]])

    from visualization import plot_route, plot_convergence
    plot_route(depot, delivery_locations, best_route.chromosome)
    plot_convergence(best_distance_history)


if __name__ == "__main__":
    main()
