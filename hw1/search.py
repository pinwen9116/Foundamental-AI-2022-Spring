# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
# Discuss: B09902029
# Reference:
# 
    # https://en.wikipedia.org/wiki/A*_search_algorithm
    # https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
    # https://github.com/rahulsk2/CS440/blob/13c6e5f4af4207209393280442a241e4dc0ed8ff/CS440%20MP1/search.py#L74
    #            heuristic method
#
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    queue = []
    s = maze.getStart()
    end = maze.getObjectives()
    if s == end[0]:
        return []
    queue.append(s)
    visited = set()
    visited.add(s)
    parent = {}
    ans = []
    while queue:
        s = queue.pop(0)      
        if s == end[0]:
            #print("END")
            break
        neighbor = maze.getNeighbors(s[0], s[1])
        for i in neighbor:
            if i not in visited and i not in queue:
                #print(i)
                queue.append(i)
                visited.add(i)
                parent[i] = s
    
    s = end[0]
    while parent[s] != maze.getStart():
        ans.append([s[0], s[1]])
        s = parent[s]
    ans.append(s)
    ans.append(parent[s])
    ans = ans[::-1]
    return ans

import queue
import sys
def manhattan(a, b):
    return (abs(a[0] - b[0]) + abs(a[1] - b[1]))
def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # Reference:
    # https://en.wikipedia.org/wiki/A*_search_algorithm
    # https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
    s = maze.getStart()
    end = maze.getObjectives()
    # node: position, parent, g, h
    open_list = []
    close_list = []
    parent = {}
    g = {}
    h = {}
    g[s] = 0
    h[s] = 0
    open_list.append(s)

    ans = []
    while len(open_list) > 0:
        now_node = open_list[0]
        now_index = 0
        
        for index, item in enumerate(open_list):
            if g[item] + h[item] < g[now_node] + h[now_node]:
                now_node = item
                now_index = index
        open_list.pop(now_index)
        close_list.append(now_node)
        if now_node == end[0]:
            while now_node is not s :
                ans.append(now_node)
                now_node = parent[now_node]
            ans.append(s)
            return ans[::-1]
            break
        
        neighbor = maze.getNeighbors(now_node[0], now_node[1])
        for i in neighbor:
            if i in close_list:
                continue
            if i not in open_list:
                parent[i] = now_node
                g[i] = g[now_node] + 1
                h[i] = manhattan(end[0], i)
                open_list.append(i)
            else:
                temp_f = g[i] + h[i]
                g[i] = min(g[i], g[now_node] + 1)
                if g[i] + h[i]< temp_f:
                    parent[i] = now_node
               
    return ans

def my_astar(maze, s, end):
    # node: position, parent, g, h
    open_list = []
    close_list = []
    parent = {}
    g = {}
    h = {}
    g[s] = 0
    h[s] = 0
    open_list.append(s)

    ans = []
    while len(open_list) > 0:
        now_node = open_list[0]
        now_index = 0
        
        for index, item in enumerate(open_list):
            if g[item] + h[item] < g[now_node] + h[now_node]:
                now_node = item
                now_index = index
        open_list.pop(now_index)
        close_list.append(now_node)
        if now_node == end:
            while now_node is not s :
                ans.append(now_node)
                now_node = parent[now_node]
            ans.append(s)
            return ans[::-1]
            break
        
        neighbor = maze.getNeighbors(now_node[0], now_node[1])
        for i in neighbor:
            if i in close_list:
                continue
            if i not in open_list:
                parent[i] = now_node
                g[i] = g[now_node] + 1
                h[i] = manhattan(end, i)
                open_list.append(i)
            else:
                temp_f = g[i] + h[i]
                g[i] = min(g[i], g[now_node] + 1)
                if g[i] + h[i]< temp_f:
                    parent[i] = now_node
               
    return ans

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    """endings = maze.getObjectives()
    starting = maze.getStart()
    paths = [[[] for _ in range(5)] for _ in range(5)]
    endings.append(starting)
    for i in range (5):
        for j in range (i+1, 5):
            paths[i][j] = astar_reuse(maze, endings[i], endings[j])
            paths[j][i] = paths[i][j][::-1]
            #print(len(paths[i][j]))

    least = 9223372036854775807

    for i in range (4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    if i != j and i != k and i != l and j != k and j != l and k != l:
                        sum = len(paths[i][j]) + len(paths[j][k]) + len(paths[k][l]) + len(paths[4][i])
                        if sum < least:
                            least = sum
                            ans = [i, j, k, l]
    ans_path = paths[4][ans[0]] + paths[ans[0]][ans[1]][1:] + paths[ans[1]][ans[2]][1:] +paths[ans[2]][ans[3]][1:]
    #ans = dijkstra(5, paths)

    return ans_path"""
    return astar_multi(maze)

len_paths = {}

def mst_h(node, goals, endings):
    if len(goals) == 0:
        return 0
    result = 0
    now_v = [endings.index(goals[0])]
    vertices = []
    for i in range(1, len(goals)):
        vertices.append(endings.index(goals[i]))
    while len(now_v) != len(goals):
        path = []
        for nv in now_v:
            minn = sys.maxsize
            mint = None
            for v in vertices:
                if v < nv:
                    edge = (v, nv)
                else:
                    edge = (nv, v)
                if len_paths[edge] < minn:
                    minn = len_paths[edge]
                    mint = v
            path.append((minn, mint))
        min_path = min(path)
        vertices.remove(min_path[1])
        result += min_path[0]
        now_v.append(min_path[1])
    ans = []
    for goal in goals:
        ans.append(manhattan(node, goal))
    return result+ min(ans)

from copy import deepcopy

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # Same as above
    
    def manhattan(a, b):
        return (abs(a[0] - b[0]) + abs(a[1] - b[1]))
    def getGoal(a, bs):
        ans = []
        for b in bs:
            if a != b:
                ans.append(b)
        return ans
    starting = maze.getStart()
    endings = maze.getObjectives()
    number = len(endings)
    #paths = [[[] for _ in range(number + 1)] for _ in range(number + 1)]
    paths = []
    for i in range(len(endings)):
        for j in range(i+1, len(endings)):
            paths.append((i, j))
    c_maze = deepcopy(maze)
    for path in paths:
        dist = len(my_astar(c_maze, endings[path[0]], endings[path[1]]))
        len_paths[path] = dist-1
    p = queue.PriorityQueue()
    parent = {(starting, tuple(endings)): None}
    s_node = (mst_h(starting, tuple(endings), endings), 0, (starting, tuple(endings)))
    p.put(s_node)
    now_node_distance_map = {s_node[2]:0}
    while p:
        now = p.get()
        now_pos = now[2][0]
        if (len(now[2][1]) == 0):
            ## 這邊break
            result = []
            fdsa = now[2]
            while fdsa != None:
                result.append(fdsa[0])
                fdsa = parent[fdsa]
            return result[::-1]

        neighbors = maze.getNeighbors(now_pos[0], now_pos[1])
        for n in neighbors:
            goals_from_node = tuple(getGoal(n, now[2][1]))
            dst_node = (n, goals_from_node)
            if dst_node in now_node_distance_map and now_node_distance_map[dst_node] <= now_node_distance_map[now[2]]+1:
                continue
            now_node_distance_map[dst_node] = now_node_distance_map[now[2]]+1
            parent[dst_node] = now[2]
            old_f = now[0]
            new_f = now_node_distance_map[dst_node]+ mst_h(n, goals_from_node, endings)
            #new_f = max(old_f, new_f)

            new_node = (new_f, now_node_distance_map[dst_node], dst_node)
            p.put(new_node)

    return []

def fast_h(node, endings):
    min_h = sys.maxsize
    for i in endings:
        if manhattan(node, i) < min_h:
            min_h = manhattan(node, i)
    if len(endings) > 2:
        return 2* min_h
    else:
        return min_h

def fast_astar(maze, starting, endings):
    p = queue.PriorityQueue()
    visited = set()
    parent = {}
    # f, distance to start point, current node
    s_node = (fast_h(starting, endings), 0, starting)
    p.put(s_node)
    ans = []
    while p:
        now = p.get()
        now_pos = now[2]
        visited.add(now_pos)
        if now_pos in endings:
            #print("does end")
            temp = now
            while temp[2] is not starting :
                ans.append(temp[2])
                temp = parent[temp[2]]
            ans.append(starting)    
            return ans[::-1]
            break
        neighbors = maze.getNeighbors(now_pos[0], now_pos[1])
        for i in neighbors:
            if i not in visited:
                parent[i] = now
                new_node = (fast_h(i, endings) + now[1] + 1, now[1] + 1, i)
                p.put(new_node)
def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # Reference: https://github.com/rahulsk2/CS440/blob/13c6e5f4af4207209393280442a241e4dc0ed8ff/CS440%20MP1/search.py#L74
    #            heuristic method
    starting = maze.getStart()
    endings = maze.getObjectives()
    ans = []
    while endings:
        res = fast_astar(maze, starting, endings)
        ans.extend(res)
        endings.remove(res[-1])
        starting = res[-1]
    return ans
