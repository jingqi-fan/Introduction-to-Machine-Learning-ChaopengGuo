import threading
import unittest

import search
from eightpuzzle import EightPuzzleState, PuzzleSearchProblem


class TimeoutException(Exception):
    pass


class TimerThread(threading.Thread):
    class InternalThread(threading.Thread):
        def __init__(self, func, args=[]):
            super(TimerThread.InternalThread, self).__init__()
            self.func = func
            self.args = args
            self._finished = False

        def run(self) -> None:
            if len(self.args) == 0:
                self.func()
            else:
                params = self.args[:]
                self.func(*params)
            self._finished = True

        def finished(self):
            return self._finished

    def __init__(self, func, args, error_message, timeout, interval=1.0):
        super(TimerThread, self).__init__()
        self.func = func
        self.args = args
        self.error_message = error_message
        self.timeout = timeout
        self.stopped = False
        self.interval = interval
        self.internal_thread = None

    def run(self):
        self.internal_thread = TimerThread.InternalThread(self.func, self.args)
        self.internal_thread.setDaemon(True)
        self.internal_thread.start()
        amount_of_interval = 0
        while not self.stopped:
            amount_of_interval += 1
            self.internal_thread.join(self.interval)
            if self.internal_thread.finished():
                self.stop()
            if 0 < self.timeout <= amount_of_interval * self.interval:
                self.stop()

    def stop(self):
        self.stopped = True
        if not self.internal_thread.finished() and self.timeout > 0:
            raise TimeoutError(self.error_message)

    def is_stop(self):
        return self.stopped


class EightPuzzleTest(unittest.TestCase):
    def depth_first_search_func(self):
        print("Starting depth first search test")
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

    def test_depth_first_search(self):
        t = TimerThread(self.depth_first_search_func, [], "Depth First Search cannot find the solution within 30s", 30)
        t.start()
        t.join()

    def breath_first_search_func(self):
        print("Starting breath first search test")
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

    def test_breath_first_search(self):
        t = TimerThread(self.breath_first_search_func, [], "Breath First Search cannot find the solution within 30s",
                        30)
        t.start()
        t.join()

    def unit_cost_search_func(self):
        print("Starting uniform cost search test")
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

    def test_unit_cost_search(self):
        t = TimerThread(self.unit_cost_search_func, [], "Uniform Cost Search cannot find the solution within 30s",
                        30)
        t.start()
        t.join()

    def a_start_search_func(self):
        print("Starting A* search test")
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

    def test_a_star_search(self):
        t = TimerThread(self.a_start_search_func, [],
                        "A* Search and Uniform Cost Search cannot find the solution within 60s",
                        60)
        t.start()
        t.join()

    def print_result(self, alg, step, cost, path):
        print(f"{alg} found a path of {len(path)} moves by {step} steps and {cost} cost")
        print(f"{path}")


if __name__ == '__main__':
    unittest.main()
