"""
Testing for the PostalDeliveryEA class

Tests:
  1. Baseline comparison  
  2. Scaling test  
  3. Hyperparameter test
"""

import math
import random

from postalRoute import PostalDeliveryEA, Location
from visualization import plot_route, plot_convergence


def generate_random_locations(num_stops: int, seed: int = 42):
    """Generate depot + random delivery stops."""
    random.seed(seed)
    depot = Location(name="Depot", x=50, y=50)
    locations = [
        Location(name=f"Stop {i}", x=random.uniform(0, 100), y=random.uniform(0, 100))
        for i in range(num_stops)
    ]
    return depot, locations


def run_ea(depot, locations, generations=300, mutation_rate=0.2):
    ea = PostalDeliveryEA(
        depot=depot,
        delivery_locations=locations,
        population_size=100,
        generations=generations,
        mutation_rate=mutation_rate,
        random_seed=42,
    )
    return ea.run()

# ----------------------------------------------------------------------
# Test 1 - Baseline Comparison
# EA should beat this strategy
# ----------------------------------------------------------------------
def nearest_neighbour_distance(depot: Location, locations: list) -> float:
    """
    Greedy nearest-neighbour heuristic where at each step, the closest 
    unvisited stop is visited.
    Returns the total route distance.
    """
    def dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    unvisited = list(range(len(locations)))
    current = depot
    total = 0.0

    while unvisited:
        nearest = min(unvisited, key=lambda i: dist(current, locations[i]))
        total += dist(current, locations[nearest])
        current = locations[nearest]
        unvisited.remove(nearest)

    total += dist(current, depot)
    return total


def test_baseline_comparison(num_stops=20):
    print("=" * 60)
    print(f"TEST 1: EA vs Greedy Baseline ({num_stops} stops)")
    print("=" * 60)

    depot, locations = generate_random_locations(num_stops)
    greedy_dist = nearest_neighbour_distance(depot, locations)
    best_route, history = run_ea(depot, locations)
    ea_dist = best_route.total_distance
    improvement = ((greedy_dist - ea_dist) / greedy_dist) * 100

    print(f"\nGreedy distance : {greedy_dist:.2f}")
    print(f"EA distance     : {ea_dist:.2f}")
    print(f"Improvement     : {improvement:.1f}%")
    print("EA beats the greedy heuristic" if ea_dist < greedy_dist else "EA did not beat greedy")

    return [depot, locations, best_route, history]


# ------------------------------------------------------------------------
# Test 2 - Scaling Test
# Runs the EA on larger and larger problems
# ------------------------------------------------------------------------
def test_scaling():
    print("\n" + "=" * 60)
    print("TEST 2: Scaling")
    print("=" * 60)

    results = []
    for num_stops in [8, 20, 50, 100]:
        depot, locations = generate_random_locations(num_stops, seed=42)
        greedy_dist = nearest_neighbour_distance(depot, locations)
        best_route, history = run_ea(depot, locations)
        ea_dist = best_route.total_distance
        improvement = ((greedy_dist - ea_dist) / greedy_dist) * 100
        results.append(([depot, locations, best_route, history], num_stops, greedy_dist, ea_dist, improvement))
        print(f"\n{num_stops:>4} stops | Greedy: {greedy_dist:>8.2f} | EA: {ea_dist:>8.2f} | Improvement: {improvement:.1f}%")

    return results


# -----------------------------------------------------------------------
# Test 3 - Hyperparameter Test
# Runs the EA with different mutation rates and compares convergence
# -----------------------------------------------------------------------
def test_mutation_rates(num_stops=20):
    print("\n" + "=" * 60)
    print(f"TEST 3: Mutation Rate Comparison ({num_stops} stops)")
    print("=" * 60)

    depot, locations = generate_random_locations(num_stops, seed=42)
    histories = {}

    for rate in [0.05, 0.1, 0.2, 0.5]:
        best_route, history = run_ea(depot, locations, mutation_rate=rate)
        histories[rate] = history
        print(f"Mutation rate {rate:.2f} -> Final distance: {best_route.total_distance:.2f}")

    return [depot, locations, best_route, histories]

if __name__ == "__main__":
    depot, locations, best_route, history = test_baseline_comparison(num_stops=20)
    plot_route(depot, locations, best_route.chromosome)
    plot_convergence(history)

    idx = 0; # idx refers to # of stops; 0=8, 1=20, 2=50, 3=100
    results = test_scaling()
    depot, locations, best_route, history = results[idx][0]
    plot_route(depot, locations, best_route.chromosome)
    plot_convergence(history)
    
    mutation = 0.05 # mutation rate to visualize; 0.05, 0.1, 0.2, or 0.5
    depot, locations, best_route, histories = test_mutation_rates(num_stops=20)
    plot_route(depot, locations, best_route.chromosome)
    plot_convergence(histories[mutation])