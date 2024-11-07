import unittest

import search
from eightpuzzle import EightPuzzleState, PuzzleSearchProblem


class EightPuzzleTest(unittest.TestCase):
    def test_depth_first_search(self):
        print("Starting depth first search test. "
              "If it cannot finish within 30s, "
              "please kill the job and review your code")
        print("---------------------------------------------")
        puzzle = EightPuzzleState([1, 0, 2, 3, 4, 5, 6, 7, 8])
        problem = PuzzleSearchProblem(puzzle)
        path, step = search.depth_first_search(problem)
        print("Test DFS on:")
        print(puzzle)

        self.print_result("DFS", step, problem.get_costs(path), path)

        curr = puzzle
        for a in path:
            curr = curr.next_state(a)
        self.assertTrue(curr.is_goal(), "The final state is not goal test")
        print("=============================================")

    def test_breath_first_search(self):
        print("Starting breath first search test",
              "If it cannot finish within 30s, "
              "please kill the job and review your code")
        print("---------------------------------------------")
        puzzle = EightPuzzleState([3, 0, 5, 6, 2, 4, 7, 8, 1])
        problem = PuzzleSearchProblem(puzzle)
        path, step = search.breadth_first_search(problem)
        print("Test BFS on: \n")
        print(puzzle)
        self.print_result("BFS", step, problem.get_costs(path), path)

        curr = puzzle
        for a in path:
            curr = curr.next_state(a)
        self.assertTrue(curr.is_goal(), "The final state is not goal test")
        print("=============================================")

    def test_unit_cost_search(self):
        print("Starting uniform cost search test"
              "If it cannot finish within 30s, "
              "please kill the job and review your code")
        print("---------------------------------------------")
        puzzle = EightPuzzleState([3, 0, 5, 6, 2, 4, 7, 8, 1])
        problem = PuzzleSearchProblem(puzzle)
        path, step = search.uniform_cost_search(problem)
        print("Test UCS on: \n")
        print(puzzle)
        self.print_result("UCS", step, problem.get_costs(path), path)

        curr = puzzle
        for a in path:
            curr = curr.next_state(a)
        self.assertTrue(curr.is_goal(), "The final state is not goal test")
        self.assertEqual(problem.get_costs(path), 43, "The answer may not the optimal one")
        print("=============================================")

    def test_a_star_search(self):
        print("Starting A* search test, "
              "If it cannot finish within 60s, "
              "please kill the job and review your code")
        print("---------------------------------------------")
        puzzle = EightPuzzleState([3, 0, 5, 6, 2, 4, 7, 8, 1])
        problem = PuzzleSearchProblem(puzzle)
        path_a_start, step_a_star = search.a_start_search(problem)
        path_ucs, step_ucs = search.uniform_cost_search(problem)
        print("Test A* on: \n")
        print(puzzle)
        self.print_result("A*", step_a_star, problem.get_costs(path_a_start), path_a_start)

        curr = puzzle
        for a in path_a_start:
            curr = curr.next_state(a)
        self.assertTrue(curr.is_goal(), "The final state is not goal test")
        self.assertEqual(problem.get_costs(path_a_start), 43, "The answer may not the optimal one")
        self.assertLessEqual(step_a_star, step_ucs, "The A* steps should be less or equal compared with UCS")
        print("=============================================")

    def print_result(self, alg, step, cost, path):
        print(f"{alg} found a path of {len(path)} moves by {step} steps and {cost} cost")
        print(f"{path}")


if __name__ == '__main__':
    unittest.main()
