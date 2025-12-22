from typing import List, Dict, Set, Tuple, Optional, Callable
import heapq
import torch

def select_neighbors(
    vec: torch.Tensor,
    vertex_ids: List[int],
    vertices: List[torch.Tensor],
    num_selections: int,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], float]
) -> List[int]:
    distances = [(distance_fn(vertices[id], vec), id) for id in vertex_ids]
    distances.sort()
    return [id for _, id in distances[:num_selections]]

class NSW:
    def __init__(
        self,
        vertices: List[torch.Tensor],
        distance_fn: Callable[[torch.Tensor, torch.Tensor], float],
        max_edges: int
    ):
        self.vertices = vertices
        self.distance_fn = distance_fn
        self.max_edges = max_edges
        self.edges: Dict[int, List[int]] = {}
        self.in_vertices: List[int] = []

    def search_layer(
        self,
        base_vector: torch.Tensor,
        limit: int,
        entry_points: List[int]
    ) -> List[int]:
        result_candidates: List[int] = []
        visited: Set[int] = set()
        to_explore: List[Tuple[float, int]] = []
        result_set: List[Tuple[float, int]] = []

        for entry_point in entry_points:
            visited.add(entry_point)
            dist = self.distance_fn(self.vertices[entry_point], base_vector)
            heapq.heappush(to_explore, (dist, entry_point))
            heapq.heappush(result_set, (-dist, entry_point))

        while to_explore:
            dist, vertex = heapq.heappop(to_explore)
            if dist > -result_set[0][0]:
                break

            for neighbor in self.edges.get(vertex, []):
                if neighbor in visited: continue
                visited.add(neighbor)

                dist = self.distance_fn(self.vertices[neighbor], base_vector)
                heapq.heappush(to_explore, (dist, neighbor))
                heapq.heappush(result_set, (-dist, neighbor))
                if len(result_set) > limit:
                    heapq.heappop(result_set)

        while result_set:
            result_candidates.append(heapq.heappop(result_set)[1])
        return result_candidates[::-1]

    def add_vertex(self, vertex_id: int) -> None:
        self.in_vertices.append(vertex_id)

    def connect(self, vertex_a: int, vertex_b: int) -> None:
        self.edges.setdefault(vertex_a, []).append(vertex_b)
        self.edges.setdefault(vertex_b, []).append(vertex_a)

    def default_entry_point(self) -> Optional[int]:
        return self.in_vertices[0] if self.in_vertices else None

    def insert(
        self,
        vec: torch.Tensor,
        vertex_id: int,
        ef_construction: int,
        num_neighbors_to_con: int
    ) -> None:
        self.add_vertex(vertex_id)

        if (len(self.in_vertices) <= 1) or (not self.default_entry_point()): 
            return

        ef_con_nodes = self.search_layer(vec, ef_construction, [self.default_entry_point()])
        neighbors = select_neighbors(vec, ef_con_nodes, self.vertices, num_neighbors_to_con, self.distance_fn)
        for neighbor in neighbors:
            self.connect(vertex_id, neighbor)
        for neighbor in neighbors:
            edges = self.edges.get(neighbor, [])
            if len(edges) > self.max_edges:
                new_neighbors = select_neighbors(self.vertices[neighbor], edges, self.vertices, self.max_edges, self.distance_fn)
                self.edges[neighbor] = new_neighbors
