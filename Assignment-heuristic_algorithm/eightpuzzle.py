import copy
import random

import search


class EightPuzzleState:
    template = """-------------
| {} | {} | {} |
| {} | {} | {} |
| {} | {} | {} |
-------------"""

    def __init__(self, numbers):
        assert len(numbers) == 9 and sorted(numbers) == list(
            range(9)), "Eight Puzzle only accept 9 numbers in which 0 presents blank hole"
        self.cells = [[], [], []]
        for i in range(3):
            for j in range(3):
                self.cells[i].append(numbers[i * 3 + j])
                if numbers[i * 3 + j] == 0:
                    self.blank_location = i, j

    def is_goal(self):
        return self.cells == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def legal_moves(self):
        moves = []
        row, col = self.blank_location
        if row != 0:
            moves.append('up')
        if row != 2:
            moves.append('down')
        if col != 0:
            moves.append('left')
        if col != 2:
            moves.append('right')
        return moves

    def next_state(self, move):
        # assert move in self.legal_moves(), "Cannot do this move"
        row, col = self.blank_location
        if move == 'up':
            newrow = row - 1
            newcol = col
        elif move == 'down':
            newrow = row + 1
            newcol = col
        elif move == 'left':
            newrow = row
            newcol = col - 1
        elif move == 'right':
            newrow = row
            newcol = col + 1
        else:
            raise Exception("Illegal Move")
        new_state = copy.deepcopy(self)
        new_state.cells[row][col], new_state.cells[newrow][newcol] = self.cells[newrow][newcol], self.cells[row][col]
        new_state.blank_location = newrow, newcol
        return new_state

    def __eq__(self, other):
        return self.cells == other.cells

    def __hash__(self):
        return hash(str(self.cells))

    def __str__(self):
        elements = list(map(
            lambda value: ' ' if value == 0 else str(value),
            sum(self.cells, [])
        ))
        return EightPuzzleState.template.format(*elements)


class PuzzleSearchProblem(search.SearchProblem):
    cost_dict = {
        "up": 0,
        "down": 10,
        "left": 1,
        "right": 3
    }
    goal_state = EightPuzzleState(list(range(9)))

    def __init__(self, start_state):
        self.start_state = start_state

    def get_start_state(self):
        return self.start_state

    def is_goal_state(self, state):
        return state.is_goal()

    def get_successor(self, state):
        legal_moves = state.legal_moves()
        return [(state.next_state(move), move, PuzzleSearchProblem.cost_dict[move]) for move in legal_moves]

    def get_costs(self, actions):
        return sum([PuzzleSearchProblem.cost_dict[action] for action in actions])

    def get_goal_state(self):
        return PuzzleSearchProblem.goal_state


def createRandomEightPuzzle(moves=100):
    puzzle = EightPuzzleState(list(range(9)))
    for i in range(moves):
        puzzle = puzzle.next_state(random.sample(puzzle.legal_moves(), 1)[0])
    return puzzle


if __name__ == '__main__':
    # puzzle = createRandomEightPuzzle(30)

    puzzle = EightPuzzleState([3, 0, 5, 6, 2, 4, 7, 8, 1])
    print('A random puzzle:')
    print(puzzle)

    problem = PuzzleSearchProblem(puzzle)
    path, step = search.uniform_cost_search(problem)
    print(('BFS found a path of %d moves: %s by %d steps cost %d' % (
        len(path), str(path), step, problem.get_costs(path))))
    curr = puzzle
    i = 1
    for a in path:
        curr = curr.next_state(a)
        print(('After %d move%s: %s' % (i, ("", "s")[i > 1], a)))
        print(curr)

        input("Press return for the next state...")  # wait for key stroke
        i += 1
