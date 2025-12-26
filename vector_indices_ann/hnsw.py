from nsw import NSW, select_neighbors
from typing import Callable, List, Tuple
import torch
import math
import random

class HNSWIndex:
    def __init__(
        self,
        distance_fn: Callable[[torch.Tensor, torch.Tensor], float],
        max_edges: int,
        ef_construction: int,
        ef_search: int,
        *,
        nsw_only=False
    ):
        self.distance_fn = distance_fn
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_edges = max_edges
        self.max_edges_0 = max_edges ** 2
        self.level_normalization = 1.0 / math.log(max_edges)
        
        self.vertices: List[torch.Tensor] = []
        self.rids: List[int] = []
        self.layers: List[NSW] = [NSW(self.vertices, distance_fn, self.max_edges_0)]
        self.nsw_only = nsw_only,
        self.generator = random.Random()

    def add_vertex(self, vec: torch.Tensor, rid: int) -> int:
        vertex_id = len(self.vertices)
        self.vertices.append(vec)
        self.rids.append(rid)
        return vertex_id

    def build_index(self, initial_data: List[Tuple[torch.Tensor, int]]):
        random.shuffle(initial_data)
        for vec, rid in initial_data:
            self.insert(vec, rid)

    def search(self, vec: torch.Tensor, limit: int) -> List[int]:
        if self.nsw_only:
            nearest_elements = self.layers[0].search_layer(vec, limit, [self.layers[0].default_entry_point()])
            return [self.rids[vid] for vid in nearest_elements]

        entry_points = [self.layers[-1].default_entry_point()]
        for level in range(len(self.layers)-1, 0, -1):
            nearest_elements = self.layers[level].search_layer(vec, self.ef_search, entry_points)
            nearest_elements = select_neighbors(vec, nearest_elements, self.vertices, 1, self.distance_fn)
            entry_points = nearest_elements

        nearest_elements = self.layers[0].search_layer(vec, max(limit, self.ef_search), entry_points)
        neighbors = select_neighbors(vec, nearest_elements, self.vertices, limit, self.distance_fn)
        return [self.rids[vid] for vid in neighbors]

    def insert(self, vec: torch.Tensor, rid: int):
        if self.nsw_only:
            vertex_id = self.add_vertex(vec, rid)
            self.layers[0].insert(vec, vertex_id, self.ef_construction, self.max_edges)
            return

        vertex_id = self.add_vertex(vec, rid)
        target_level = math.floor(-math.log(self.generator.random()) * self.level_normalization)
        target_level = max(0, target_level)
        
        nearest_elements: List[int] = []
        
        if self.layers[0].in_vertices:
            entry_points = [self.layers[-1].default_entry_point()]
            level = len(self.layers) - 1
            
            while level > target_level:
                nearest_elements = self.layers[level].search_layer(vec, self.ef_search, entry_points)
                nearest_elements = select_neighbors(vec, nearest_elements, self.vertices, 1, self.distance_fn)
                entry_points = nearest_elements
                level -= 1
            
            while level >= 0:
                layer = self.layers[level]
                nearest_elements = layer.search_layer(vec, self.ef_construction, entry_points)
                neighbors = select_neighbors(vec, nearest_elements, self.vertices, self.max_edges, self.distance_fn)
                
                layer.add_vertex(vertex_id)
                for neighbor in neighbors:
                    layer.connect(vertex_id, neighbor)
                
                for neighbor in neighbors:
                    edges = layer.edges.get(neighbor, [])
                    if len(edges) > (self.max_edges_0 if level == 0 else self.max_edges):
                        max_edges = self.max_edges_0 if level == 0 else self.max_edges
                        new_neighbors = select_neighbors(
                            self.vertices[neighbor],
                            edges,
                            self.vertices,
                            max_edges,
                            self.distance_fn
                        )
                        layer.edges[neighbor] = new_neighbors
                
                entry_points = nearest_elements
                level -= 1
        else:
            self.layers[0].add_vertex(vertex_id)
        
        while len(self.layers) <= target_level:
            new_layer = NSW(self.vertices, self.distance_fn, self.max_edges)
            new_layer.add_vertex(vertex_id)
            self.layers.append(new_layer)
