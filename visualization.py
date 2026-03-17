import matplotlib.pyplot as plt

# Plot route visit order
def plot_route(depot, locations, chromosome, title="Route"):
    # start at origin
    x_route = [depot.x]
    y_route = [depot.y]

    # trace route
    for i in chromosome:
        x_route.append(locations[i].x)
        y_route.append(locations[i].y)
    
    # end at origin
    x_route.append(depot.x)
    y_route.append(depot.y)

    # plot route
    plt.figure(figsize=(8, 6))
    plt.plot(x_route, y_route, marker='o', linestyle='-')

    # plot depot
    plt.scatter(depot.x, depot.y, c='red', s=100, label='Depot')
    plt.text(depot.x, depot.y, "Depot", fontsize=10, ha='right')

    # plot visit order
    for step, idx in enumerate(chromosome):
        loc = locations[idx]
        plt.text(loc.x, loc.y, f"{step+1}", fontsize=10, ha='right')

    plt.title(title)
    plt.xlabel("X Pos.")
    plt.ylabel("Y Pos.")
    plt.grid(True)
    plt.show()

# Plot convergence over generations
def plot_convergence(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history, color='blue')
    plt.title("EA Convergence (Route Distance Over Generations)")
    plt.xlabel("Generation #")
    plt.ylabel("Best Route Distance")
    plt.grid(True)
    plt.show()