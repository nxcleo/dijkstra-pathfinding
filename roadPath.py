import copy


class Graph:
    def __init__(self):
        """
        constructor for the graph class

        adjacency_list: a list of lists of edges (i.e. [[Edge, Edge ..], [Edge, Edge..] ..]),
        such that the element at adjacency_list[i] is the list of all adjacent edges of node i.
        Each edge is an Edge object containing u, v, and weight.

        adjacency_list_aug: similar but augmented for task 2

        services: list of service nodes.
        """
        self.adjacency_list = []
        self.adjacency_list_aug = []
        self.services = []

    def buildGraph(self, f_name):
        """
        Creates the adjacency matrix for the Graph class

        1. read all the edges from file: space and time complexity of O(E)
        2. find the max vertex from edge list, O(1) space and O(E) time
        3. create adjacency list from edge list, space and time complexity of O(E)

        overall complexity: space and time complexity of O(E)
        :param f_name: Name of file containing edges
        :return: modify self.adjacency_list
        """

        edges = []
        with open(f_name) as file:
            for line in file:
                line = line.strip().split(" ")
                edges.append([int(line[0]), int(line[1]), float(line[2])])

        max_vertex = 0
        for edge in edges:
            if edge[0] > max_vertex:
                max_vertex = edge[0]
            if edge[1] > max_vertex:
                max_vertex = edge[1]

        self.adjacency_list = [[] for _ in range(max_vertex + 1)]

        for edge in edges:
            self.adjacency_list[edge[0]].append(Edge(edge[0], edge[1], edge[2]))

    def quickestPath(self, source, target, full=False):
        """
        Use Dijkstra's algorithm to find shortest path from source to target.
        uses a min heap to maintain list of discovered nodes
        discovered saves a tuple for each node: (node, distance to source, previous node)
        finalised list stores information for node i at index i, each contains the distance to source and previous node

        overall time complexity: O(E log(V))
        overall space complexity: O(E + V)
        :param source:  starting node
        :param target:  target node
        :param full:    if full is true, algorithm will continue until discovered is empty,
                        if False, it will stop when target is finalised
        :return:        path and the length of path
        """
        finalised = [None for _ in range(len(self.adjacency_list))]
        discovered = MinHeap(len(self.adjacency_list))
        discovered.insert((source, 0, None))

        while len(discovered) > 0:
            current, dist, last = discovered.pop()
            if finalised[current] is not None:
                continue

            finalised[current] = [dist, last]
            if current == target and not full:
                break
            # print(current, dist, last)
            for edge in self.adjacency_list[current]:
                if discovered.vertex_index[edge.v] is None:
                    discovered.insert((edge.v, dist + edge.w, current))
                else:
                    discovered.update((edge.v, dist + edge.w, current))

        if full:
            return finalised

        if current == target:
            cost = dist
            path = [current]
            while finalised[current][1] is not None:
                path.append(finalised[current][1])
                current = finalised[current][1]
            return path[::-1], cost
        else:
            return [[], -1]

    def augmentGraph(self, filename_camera, filename_toll):
        """
        creates a new adjacency list: adjacency_list_aug which is a copy of the original from task 1,
        the nodes with camera are not to be visited, so it is set to have no edges
        path with toll are to be avoided, so they are deleted.

        Space complexity: O(V + E)
        Time complexity for cameras:    O(V)
        Time complexity for tolls:      O(E^2)

        """

        self.adjacency_list_aug = copy.deepcopy(self.adjacency_list)

        with open(filename_camera) as cameras:
            for camera in cameras:
                camera = camera.strip()
                self.adjacency_list_aug[int(camera)] = []

        with open(filename_toll) as tolls:
            for toll in tolls:
                toll = toll.strip().split(" ")
                for i in range(len(self.adjacency_list_aug[int(toll[0])])):
                    if self.adjacency_list_aug[int(toll[0])][i].v == toll[1]:
                        self.adjacency_list_aug[int(toll[0])][i], self.adjacency_list_aug[int(toll[0])][-1] = \
                            self.adjacency_list_aug[int(toll[0])][-1], self.adjacency_list_aug[int(toll[0])][i]
                        del self.adjacency_list_aug[int(toll[0])][-1]


    def quickestSafePath(self, source, target):
        """
        finds the shortest path without cameras and toll
        set the adjacency_list to adjacency_list_aug, and run the quickestPath function
        reset the adjacency_list

        overall time complexity: O(E log(V))
        overall space complexity: O(E + V)
        :param source:  starting node
        :param target:  target node
        :return:        path and the length of path
        """
        temp = self.adjacency_list
        self.adjacency_list = self.adjacency_list_aug
        result = self.quickestPath(source, target)
        self.adjacency_list = temp
        return result

    def addService(self, filename_service):
        """
        adds the service nodes in the file to the services list
        :param filename_service: file name
        """
        with open(filename_service) as services:
            for service in services:
                service = service.strip()
                self.services.append(int(service))

    def quickestDetourPath(self, source, target):
        """
        finds the shortest path visiting at least one detour node
        overall time complexity: O(E log(V))
        overall space complexity: O(E + V)
        :param source: source node
        :param target: target node
        :return:
        """

        finalised = self.quickestPath(source, target, True)
        self.adjacency_list.append([])
        for service in self.services:
            if finalised[service] is not None:
                self.adjacency_list[-1].append(Edge(len(self.adjacency_list)-1, service, finalised[service][0]))
        path2, cost = self.quickestPath(len(self.adjacency_list)-1, target)

        if cost != -1:
            current = path2[1]
            path = [current]
            while finalised[current][1] is not None:
                path.append(finalised[current][1])
                current = finalised[current][1]
            return path[::-1] + path2[2:], cost
        else:
            return [[], -1]





class Edge:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        return str(self.u)+"--"+str(self.w)+"->"+str(self.v)


class MinHeap:
    def __init__(self, n):
        """
        Constructor for min heap
        array:          base array for the heap
        vertex_index:   records the location of a vertex in the heap array
        length:         records the number of items currently in the heap, used to manage deleted items
        time complexity: O(V)
        Space complexity: O(V)

        :param n: number of vertices for creating the vertex index list
        """
        self.array = []
        self.vertex_index = [None]*n
        self.length = 0

    def __len__(self):
        return self.length

    def swap(self, i, j):
        """
        swap item i and item j in the heap array, also maintains the vertex index list
        :param i:
        :param j:
        :return:
        """
        vertex_i = self.array[i][0]
        vertex_j = self.array[j][0]
        self.vertex_index[vertex_i], self.vertex_index[vertex_j] = self.vertex_index[vertex_j], self.vertex_index[vertex_i]
        self.array[i], self.array[j] = self.array[j], self.array[i]

    def float(self, pos=-1):
        """
        Float the last inserted element to the appropriate location in heap such that heap property is maintained.
        Order determined by second item (# of word occurrence), if second items are the same, the item that entered later
        (meaning it is lexicographically larger) will be pushed up further so it is of lower priority.

        Time complexity:
        O(len(x)) = O(x)
        O(h_float(k)) = O(log(k))

        Space Complexity:
        O(km), Auxiliary Space: O(1)

        :param pos: position of the out of place item, last item by default
        :return: modifies in place
        """
        if pos == -1:
            pos = len(self) - 1
        while pos > 0 and self.array[(pos - 1) // 2][1] > self.array[pos][1]:
            self.swap((pos - 1) // 2, pos)
            pos = (pos - 1) // 2

    def min_child(self, i):
        """
        Get min child node index of parent at index i in the given heap.

        Time complexity:
        O(len(x)) = O(x)
        O(h_min_child(k)) = O(1)

        Space Complexity:
        O(km), Auxiliary Space: O(1)
        :param heap:    list: min heap
        :param i:       int: index of parent
        :return:        int: index of min child
        """
        c1 = self.array[i*2+1][1] if i*2+1 < self.length else float("inf")
        c2 = self.array[i*2+2][1] if i*2+2 < self.length else float("inf")
        if c1 < c2:
            return i*2+1
        elif c2 < c1:
            return i*2+2
        elif c1 == float("inf"):
            return -1
        else:
            return i*2+1

    def sink(self):
        """
        Sink the root node down to the appropriate location in heap such that heap property is maintained.
        Order determined by second item (# of word occurrence), if second items are the same, the item that entered later
        (meaning it is lexicographically larger) will be considered to be smaller.

        Time complexity:
        O(h_min_child(k)) = O(1)
        O(h_sink(k)) = O(log(k))

        Space Complexity:
        O(km), Auxiliary Space: O(1)

        :param heap: list in min heap form where the root node may be out of place
        :return: modifies in place
        """
        pos = 0
        min_child = self.min_child(pos)
        while min_child != -1 and self.array[pos][1] > self.array[min_child][1]:
            self.swap(pos, min_child)
            pos = min_child
            min_child = self.min_child(pos)

    def insert(self, item):
        """
        adds an item into the heap
        Time complexity: O(log(V))
        Space complexity: O(1)
        :param item: tuple to add
        """
        if len(self.array) > self.length:
            self.array[self.length] = item
        else:
            self.array.append(item)
        self.vertex_index[item[0]] = self.length
        self.length += 1
        self.float()

    def pop(self):
        """
        returns the smallest item in the heap and update heap structure
        Time complexity: O(log(V))
        Space complexity: O(1)
        :return:
        """
        res = self.array[0]
        self.swap(0, self.length-1)
        self.length -= 1
        self.vertex_index[res[0]] = None
        self.sink()
        return res

    def update(self, item):
        """

        Time complexity: O(log(V))
        Space complexity: O(1)
        """
        idx = self.vertex_index[item[0]]
        if item[1] < self.array[idx][1]:
            self.array[idx] = item
            self.float(idx)


if __name__ == '__main__':
    g = Graph()
    g.buildGraph('basicGraph.txt')
    g.addService('service.txt')

    print(g.quickestPath(1,6))