
from generator_problem import GeneratorProblem


class Solve:

    def __init__(self, n_generator, n_device, seed):

        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)

    def solve_naive(self):

        print("Solve with a naive algorithm")
        print("All the generators are opened, and the devices are associated to the closest one")

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
        self.instance.plot_solution(assigned_generators, opened_generators)

        # print("[ASSIGNED-GENERATOR]", assigned_generators)
        # print("[OPENED-GENERATOR]", opened_generators)
        # print("[SOLUTION-COST]", total_cost)

        return opened_generators,assigned_generators,total_cost

    def solve(self):
        # solution initial
        best_opened_generators, best_assigned_generators, best_cost = self.solve_naive()
        temp_assigned_generators = best_assigned_generators
        temp_opened_generators = best_opened_generators
        temp_cost = best_cost

        for iterations in range(10):
            for i in range(self.n_generator):
                assigned_generators = best_assigned_generators
                opened_generators = best_opened_generators
                opened_generators[i] = 0

                for j in range(self.n_device):
                    if assigned_generators[j] == i:
                        gen_index = range(self.n_generator)
                        gen_index = gen_index[:i-1] + gen_index[i+1:]
                        closest_generator = min(gen_index,
                                                key=lambda k: self.instance.get_distance(self.instance.device_coordinates[j][0],
                                                                                         self.instance.device_coordinates[j][1],
                                                                                         self.instance.generator_coordinates[k][0],
                                                                                         self.instance.generator_coordinates[k][1])
                                                )

                        assigned_generators[j] = closest_generator
                cost = self.instance.get_solution_cost(best_assigned_generators, best_opened_generators)

                if cost <= best_cost:
                    temp_assigned_generators = assigned_generators
                    temp_opened_generators = opened_generators
                    temp_cost = cost
                else:
                    opened_generators[i] = 1

            best_assigned_generators = temp_assigned_generators
            best_opened_generators = temp_opened_generators
            best_cost = temp_cost

        self.instance.solution_checker(best_assigned_generators, best_opened_generators)
        self.instance.plot_solution(best_assigned_generators, best_opened_generators)
        print("[ASSIGNED-GENERATOR]", best_assigned_generators)
        print("[OPENED-GENERATOR]", best_opened_generators)
        print("[SOLUTION-COST]", best_cost)






