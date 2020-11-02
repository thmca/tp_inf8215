import math
import time

from solve import Solve


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--n_generator', type=int, default=25)
    parser.add_argument('--n_device', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()


start = time.time()

if __name__ == '__main__':

    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] EVALUATING THE GENERATOR PROBLEM")
    print("[INFO] n_generator: %d" % args.n_generator)
    print("[INFO] n_device': %d" % args.n_device)
    print("[INFO] seed: %s" % args.seed)
    print("***********************************************************")

    solveur = Solve(args.n_generator, args.n_device, args.seed)

    # Solution initiale avec les noeuds assignes a la centrale la plus proche et iterations sur les centrales sans composante aleatoire, 500 iterations
    best_assigned_generators, best_opened_generators, best_cost = solveur.solve(100, False, False)

    # Solution initiale avec les noeuds assignes a la centrale la plus proche et iterations sur les centrales avec composante aleatoire, 5000 iterations
    for restart in range(25):
        random_assigned_generators, random_opened_generators, random_cost = solveur.solve(1000, True, False)
        if random_cost < best_cost:
            best_assigned_generators, best_opened_generators, best_cost = random_assigned_generators, random_opened_generators, random_cost


    # Solution initiale aleatoire et iterations sur les centrales avec composante aleatoire, 5000 iterations
    for restart in range(25):
        random_assigned_generators, random_opened_generators, random_cost = solveur.solve(5000, True, True)
        if random_cost < best_cost:
            best_assigned_generators, best_opened_generators, best_cost = random_assigned_generators, random_opened_generators, random_cost

    solveur.instance.solution_checker(best_assigned_generators, best_opened_generators)
    solveur.instance.plot_solution(best_assigned_generators, best_opened_generators)
    print("[ASSIGNED-GENERATOR]", best_assigned_generators)
    print("[OPENED-GENERATOR]", best_opened_generators)
    print("[SOLUTION-COST]", best_cost)

end = time.time()
elapsed = end-start
print("Executioon time : ", str(math.floor(elapsed/60)) + ":" + str(format(elapsed % 60, '.2f')))
