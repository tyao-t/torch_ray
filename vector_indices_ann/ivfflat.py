from typing import Callable, List, Tuple
import torch
from torch import Tensor
import random

class IVFFlatIndex:
    def __init__(
        self, 
        distance_fn: Callable[[Tensor, Tensor], float], 
        num_centroids: int, 
        num_probes: int
    ):
        self.distance_fn: Callable[[Tensor, Tensor], float] = distance_fn
        self.num_centroids: int = num_centroids
        self.num_probes: int = num_probes
        self.centroids: List[Tensor] = []
        self.clusters: List[List[Tuple[Tensor, int]]] = []

    def find_nearest_centroid(self, vec: Tensor) -> int:
        distances: List[float] = [self.distance_fn(vec, c) for c in self.centroids]
        return min(range(len(distances)), key=distances.__getitem__)

    def build_index(self, data: List[Tuple[Tensor, int]], num_start_iters=500) -> None:
        if not data:
            return
            
        random.shuffle(data)
        self.centroids = [vec.clone() for vec, _ in data[:self.num_centroids]]
        self.clusters = [[] for _ in range(self.num_centroids)]
        
        for _ in range(num_start_iters):
            new_clusters: List[List[Tuple[Tensor, int]]] = [[] for _ in range(self.num_centroids)]
            for vec, rid in data:
                centroid_idx: int = self.find_nearest_centroid(vec)
                new_clusters[centroid_idx].append((vec, rid))
            for i in range(self.num_centroids):
                if new_clusters[i]:
                    self.centroids[i] = torch.mean(torch.stack([v for v, _ in new_clusters[i]]), dim=0)
            self.clusters = new_clusters

    def search(self, query: Tensor, top_k: int) -> List[int]:
        if not self.centroids:
            return []
        centroid_distances: List[Tuple[int, float]] = [
            (i, self.distance_fn(query, c)) for i, c in enumerate(self.centroids)
        ]
        centroid_distances.sort(key=lambda x: x[1])
        candidates: List[Tuple[Tensor, int]] = []
        for i, _ in centroid_distances[:self.num_probes]:
            candidates.extend(self.clusters[i])
        candidates.sort(key=lambda x: self.distance_fn(query, x[0]))
        return [rid for _, rid in candidates[:top_k]]

    def insert(self, vector: Tensor, rid: int) -> None:
        if not self.centroids:
            raise RuntimeError("Index not built")
        centroid_idx: int = self.find_nearest_centroid(vector)
        self.clusters[centroid_idx].append((vector, rid))
