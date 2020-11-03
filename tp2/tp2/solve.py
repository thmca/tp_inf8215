import random

from generator_problem import GeneratorProblem


class Solve:

    def __init__(self, n_generator, n_device, seed):

        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)

    def solve_naive(self):

        # print("Solve with a naive algorithm")
        # print("All the generators are opened, and the devices are associated to the closest one")

        opened_generators = [1 for _ in range(self.n_generator)]

        assigned_generators = [None for _ in range(self.n_device)]

        for i in range(self.n_device):
            closest_generator = min(range(self.n_generator),
                                    key=lambda j: self.instance.get_distance(self.instance.device_coordinates[i][0],
                                                                             self.instance.device_coordinates[i][1],
                                                                             self.instance.generator_coordinates[j][0],
                                                                             self.instance.generator_coordinates[j][1])
                                    )

            assigned_generators[i] = closest_generator

        self.instance.solution_checker(assigned_generators, opened_generators)
        total_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)

        return opened_generators, assigned_generators, total_cost

    def solve_random(self):
        opened_generators = [1 for _ in range(self.n_generator)]
        assigned_generators = [random.randint(0, 24) for _ in range(self.n_device)]
        total_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)

        return opened_generators, assigned_generators, total_cost

    def solve(self, randomizeDecision, randomizeInitialSolution):
        # solution initiale
        best_opened_generators, best_assigned_generators, best_cost = self.solve_random() if randomizeInitialSolution else self.solve_naive()
        temp_assigned_generators = best_assigned_generators.copy()
        temp_opened_generators = best_opened_generators.copy()
        temp_cost = best_cost

        worseSolutions = 0

        while True:
            if worseSolutions == 25:
                break
            worseSolutions= 0
            for i in range(self.n_generator):
                assigned_generators = best_assigned_generators.copy()
                opened_generators = best_opened_generators.copy()

                # ouvrir le generateur concerne s'il est ferme et le fermer s'il est ouvert
                opened_generators[i] = int(opened_generators[i] == 0)
                if sum(opened_generators) == 0:
                    return best_assigned_generators, best_opened_generators, best_cost

                for j in range(self.n_device):
                    if assigned_generators[j] == i:
                        generators_indexes = [a for a, x in enumerate(opened_generators) if x == 1]
                        closest_generator = min(generators_indexes,
                                                key=lambda k: self.instance.get_distance(
                                                    self.instance.device_coordinates[j][0],
                                                    self.instance.device_coordinates[j][1],
                                                    self.instance.generator_coordinates[k][0],
                                                    self.instance.generator_coordinates[k][1])
                                                )

                        assigned_generators[j] = closest_generator

                cost = self.instance.get_solution_cost(assigned_generators, opened_generators)

                randomDecision = bool(random.randint(0, 1)) if randomizeDecision else True

                if (cost < temp_cost):
                    if randomDecision:
                        temp_assigned_generators = assigned_generators.copy()
                        temp_opened_generators = opened_generators.copy()
                        temp_cost = cost
                else:
                    worseSolutions += 1

                # switch back
                opened_generators[i] = int(opened_generators[i] == 0)

            best_assigned_generators = temp_assigned_generators.copy()
            best_opened_generators = temp_opened_generators.copy()
            best_cost = temp_cost

        self.instance.solution_checker(best_assigned_generators, best_opened_generators)
        print("[ASSIGNED-GENERATOR]", best_assigned_generators)
        print("[OPENED-GENERATOR]", best_opened_generators)
        print("[SOLUTION-COST]", best_cost)

        return best_assigned_generators, best_opened_generators, best_cost
