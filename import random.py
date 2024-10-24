import random
from collections import Counter


def draw_balls_from_urn():
    # Initialize the urn with 5 red balls, 3 orange balls, and 1 blue ball
    urn = ["red"] * 5 + ["orange"] * 3 + ["blue"]

    # Draw two balls without replacement
    drawn_balls = random.sample(urn, 2)

    return tuple(drawn_balls)


def simulate_draws(num_simulations):
    outcomes = []
    orange_counts = []

    for _ in range(num_simulations):
        drawn_balls = draw_balls_from_urn()
        outcomes.append(drawn_balls)

        # Count the number of orange balls in the drawn balls
        orange_count = drawn_balls.count("orange")
        orange_counts.append(orange_count)

    return outcomes, orange_counts


def calculate_expectation_and_variance(orange_counts):
    # Calculate E(X)
    num_simulations = len(orange_counts)
    E_X = sum(orange_counts) / num_simulations

    # Calculate E(X^2)
    E_X2 = sum((x**2) for x in orange_counts) / num_simulations

    # Calculate V(X)
    V_X = E_X2 - E_X**2

    return E_X, V_X


# Run the simulation 1000 times
num_simulations = 1000
outcomes, orange_counts = simulate_draws(num_simulations)

# Count the frequency of each outcome
outcome_counts = Counter(outcomes)

# Print the results
print("Outcomes and their frequencies over 1000 simulations:")
for outcome, count in outcome_counts.items():
    print(f"{outcome}: {count} times")

# Calculate E(X) and V(X)
E_X, V_X = calculate_expectation_and_variance(orange_counts)
print(
    f"\nExpected number of orange balls drawn (E(X)) over {num_simulations} simulations: {E_X:.4f}"
)
print(
    f"Variance of the number of orange balls drawn (V(X)) over {num_simulations} simulations: {V_X:.4f}"
)

# Example to show specific counts, if needed
# To convert the tuples into sorted tuples (since ('red', 'orange') and ('orange', 'red') are the same):
sorted_outcomes = [tuple(sorted(outcome)) for outcome in outcomes]
sorted_counts = Counter(sorted_outcomes)

print(
    "\nSimplified outcomes (ignoring draw order) and their frequencies over 1000 simulations:"
)
for outcome, count in sorted_counts.items():
    print(f"{outcome}: {count} times")
