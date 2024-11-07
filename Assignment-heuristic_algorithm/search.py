from abc import ABCMeta, abstractmethod

import util


class SearchProblem(metaclass=ABCMeta):
    @abstractmethod
    def get_start_state(self):
        pass

    @abstractmethod
    def is_goal_state(self, state):
        pass

    @abstractmethod
    def get_successor(self, state):
        # return (next_state, action, cost)
        pass

    @abstractmethod
    def get_costs(self, actions):
        pass

    @abstractmethod
    def get_goal_state(self):
        pass


class Node:
    def __init__(self, state, path=[], priority=0):
        self.state = state
        self.path = path
        self.priority = priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __lt__(self, other):
        return self.priority < other.priority


def search(problem, fringe, calc_heuristic=None, heuristic=None):
    """
    This is an simple abstracted graph search algorithm. You could
    using different combination of fringe storage, calc_heuristic, heuristic
    to implement different search algorithm.

    For example:
    LIFO Queue(Stack), None, None -> Depth First Search
    FIFO Queue, None, None -> Breadth First Search
    PriorityQueue, cost compute function, None -> Uniform Cost Search

    In order to avoid infinite graph/tree problem we setup a list (visited) to
    avoid expanding the same node.

    hint: please check the node first before expanding:

    if node.state not in visited:
        visited.append(node.state)
    else:
        continue

    hint: you could get the successor by problem.get_successor method.

    hint: for fringe you may want to use
        fringe.pop  get a node from the fringe
        fringe.push   put a node into the fringe
        fringe.empty  check whether a fringe is empty or not. If the fringe is empty this function return True
        problem.is_goal_state check whether a state is the goal state
        problem.get_successor get all successor from current state
            return value: [(next_state, action, cost)]
    """
    # Initialise Frontier Using Initial State od Problem
    start_node = Node((problem.get_start_state(), [], 0))
    if isinstance(fringe, util.Stack) or isinstance(fringe, util.Queue):
        fringe.push(start_node)
    else:
        fringe.push(start_node, 0)

    visited = []
    step = 0
    path = []  # My code

    while not fringe.empty():
        "*** YOUR CODE HERE ***"
        if len(visited) == 0:  # This block only executes the first time when visited is empty
            # Check if start node is goal.
            if problem.is_goal_state(start_node.state[0]):
                return [], 0

            # Push initial successors into fringe
            for successor in problem.get_successor(problem.get_start_state()):
                new_node = Node(successor, path)
                if isinstance(fringe, util.Stack) or isinstance(fringe, util.Queue):
                    fringe.push(new_node)
                else:
                    if calc_heuristic is not None:
                        new_node.priority = calc_heuristic(node=start_node, successor=new_node.state)
                    else:
                        new_node.priority = heuristic(problem=problem, node=start_node, successor=new_node.state)
                    fringe.push(new_node, new_node.priority)
            visited.append(start_node.state[0])

        leaf_node = fringe.pop()
        step += 1

        # Check goal state
        if problem.is_goal_state(leaf_node.state[0]):
            cur = leaf_node
            while cur.__class__.__name__ != 'list':
                path.append(cur.state[1])
                cur = cur.path
            path.reverse()
            return path, step

        # Process successors
        if leaf_node.state[0] not in visited:
            visited.append(leaf_node.state[0])
            for successor in problem.get_successor(leaf_node.state[0]):
                new_node = Node(successor, leaf_node)
                if new_node.state[0] not in visited:
                    if isinstance(fringe, util.Stack) or isinstance(fringe, util.Queue):
                        fringe.push(new_node)
                    else:
                        new_priority = calc_heuristic(node=leaf_node,
                                                      successor=new_node.state) if calc_heuristic else heuristic(
                            problem=problem, node=leaf_node, successor=new_node.state)
                        fringe.push(new_node, new_priority)
        "*** END YOUR CODE HERE ***"
    return []  # no path is found


def a_start_heuristic(problem, current_state):
    h = 0
    "*** YOUR CODE HERE ***"
    goal_state = problem.get_goal_state().cells
    current_state = current_state.cells
    # print(goal_state)
    # print(current_state)
    for i in range(0, 2):
        for j in range(0, 2):
            if goal_state[i][j] != current_state[i][j] and current_state[i][j != 0]:
                h = h + manhattan_distance([i, j], find_number(current_state[i][j]))
    "*** END YOUR CODE HERE ***"
    return h


def find_number(number):
    return {
        1: [0, 1],
        2: [0, 2],
        3: [2, 0],
        4: [2, 1],
        5: [2, 2],
        6: [3, 0],
        7: [3, 1],
        8: [3, 2],
    }.get(number, 'error')


def manhattan_distance(xy1, xy2):
    """Returns the manhattan distance between points xy1 and xy2"""
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def a_start_cost(problem, successor, node, heuristic=None):
    cost = 0
    "*** YOUR CODE HERE ***"
    cur = node
    while cur:
        # print(cur.__dict__)
        # print(cur.__class__.__name__)
        cost = cost + cur.state[2]
        cur = cur.path
    heuristic = a_start_heuristic(problem=problem, current_state=successor[0])
    cost = cost + heuristic
    "*** END YOUR CODE HERE ***"
    return cost


def a_start_search(problem):
    path = []
    step = 0
    "*** YOUR CODE HERE ***"
    # TODO a_start_search
    "*** END YOUR CODE HERE ***"
    return search(problem=problem, fringe=util.PriorityQueue(), heuristic=a_start_cost)
    # return path, step


def ucs_compute_node_cost(node, successor, problem=None, heuristic=None):
    """
    Define the method to compute cost within unit cost search
    hint: successor = (next_state, action, cost).
    however the cost for current node should be accumulative
    problem and heuristic should not be used by this function
    """
    cost = 0
    "*** YOUR CODE HERE ***"
    cur = node
    while cur:
        # print(cur.__dict__)
        # print(cur.__class__.__name__)
        cost = cost + cur.state[2]
        cur = cur.path
    cost = cost + successor[2]
    # print(cost)
    "*** END YOUR CODE HERE ***"
    return cost


def uniform_cost_search(problem):
    """
    Search the solution with minimum cost.
    """
    return search(problem=problem, fringe=util.PriorityQueue(), calc_heuristic=ucs_compute_node_cost)


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    hint: using util.Queue as the fringe
    """
    # path = []
    # step = 0
    "*** YOUR CODE HERE ***"
    "*** END YOUR CODE HERE ***"
    # return path, step
    return search(problem=problem, fringe=util.Queue())


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.get_start_state()
    print "Is the start a goal?", problem.is_goal_state(problem.get_start_state())
    print "Start's successors:", problem.get_successor(problem.get_start_state())

    hint: using util.Stack as the fringe
    """
    path = []
    step = 0
    "*** YOUR CODE HERE ***"
    "*** END YOUR CODE HERE ***"
    return search(problem=problem, fringe=util.Stack())
    # return path, step
