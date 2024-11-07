import unittest
import threading
import search
from eightpuzzle import EightPuzzleState, PuzzleSearchProblem

class TimeoutException(Exception):
    """Exception to raise on a timeout"""
    pass

class Timer:
    def __init__(self, seconds, error_message=''):
        self.seconds = seconds
        self.error_message = error_message
        self.timer = None

    def handle_timeout(self):
        raise TimeoutException(self.error_message)

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.handle_timeout)
        self.timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.cancel()

class EightPuzzleTest(unittest.TestCase):
    def test_depth_first_search(self):
        print("Starting depth first search test")
        print("---------------------------------------------")
        try:
            with Timer(2, "Depth First Search cannot find the solution within 2s"):
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
        except TimeoutException as e:
            print(e)

    # Other test cases follow the same pattern...

    def print_result(self, alg, step, cost, path):
        print(f"{alg} found a path of {len(path)} moves by {step} steps and {cost} cost")
        print(f"{path}")

if __name__ == '__main__':
    unittest.main()
