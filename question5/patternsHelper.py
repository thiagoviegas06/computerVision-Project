from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import math
import cv2 as cv
import numpy as np


Vec2 = Tuple[float, float]

@dataclass
class Node:
    label: str
    position: Vec2
    size: float = 0.0  
    # neighbor -> (unit_dx, unit_dy, length, angle)
    links: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)

    def add_link(self, other: "Node"):
        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]
        L  = math.hypot(dx, dy)
        if L < 1e-9: return
        ux, uy = dx/L, dy/L
        ang = math.atan2(dy, dx)
        self.links[other.label] = (ux, uy, L, ang)

@dataclass
class Pattern:
    name: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # undirected (a,b)

    def add_node(self, label: str, pos: Vec2, size: float = 0.0): # <-- UPDATE THIS METHOD
        self.nodes[label] = Node(label, pos, size=size)

    def add_edge(self, a: str, b: str):
        if a == b or (a, b) in self.edges or (b, a) in self.edges: return
        self.edges.append((a, b))
        self.nodes[a].add_link(self.nodes[b])
        self.nodes[b].add_link(self.nodes[a])

    def getHighestDegreeNode(self) -> Node:
        best = None
        best_deg = -1
        for node in self.nodes.values():
            deg = len(node.links)
            if deg > best_deg:
                best = node
                best_deg = deg
        return best
    
    def get_highest_degree_nodes(self) -> List[Node]:
        """
        Finds all nodes with the maximum number of links (degree).
        Returns a list of nodes.
        """
        if not self.nodes:
            return []

        best_deg = -1
        # First pass: find the maximum degree in the pattern
        for node in self.nodes.values():
            best_deg = max(best_deg, len(node.links))
        
        if best_deg == -1:
            return []

        # Second pass: collect all nodes that have this maximum degree
        highest_degree_nodes = [
            node for node in self.nodes.values() if len(node.links) == best_deg
        ]
        return highest_degree_nodes

def extract_pattern_from_image(bgr_or_rgba, name="pattern",
                               min_node_area=5, max_node_area=5_000,
                               green_hsv=((35, 40, 40), (90, 255, 255)),
                               white_hsv=((0, 0, 200), (180, 50, 255)),
                               link_hit_thresh=0.75,   # ≥ 75% of samples must be green
                               max_link_gap=5,         # max consecutive non-green samples
                               max_pair_px=99999       # (optionally limit how far nodes can connect)
                               ) -> Pattern:
    img = bgr_or_rgba
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    elif img.shape[2] == 3:
        bgr = img
        alpha = None
    else:
        raise ValueError("Input image must have 3 (BGR) or 4 (BGRA) channels")

    # HSV segmentation
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    white = cv.inRange(hsv, white_hsv[0], white_hsv[1])
    green = cv.inRange(hsv, green_hsv[0], green_hsv[1])

    # Clean masks a bit
    white = cv.morphologyEx(white, cv.MORPH_OPEN,  np.ones((3,3), np.uint8))
    white = cv.morphologyEx(white, cv.MORPH_DILATE, np.ones((3,3), np.uint8))
    green = cv.morphologyEx(green, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # Nodes from connected components
    num, labels, stats, cents = cv.connectedComponentsWithStats(white, connectivity=8)
    node_data = []
    for i in range(1, num):
        area = stats[i, cv.CC_STAT_AREA]
        if min_node_area <= area <= max_node_area:
            cx, cy = cents[i]  # float (x,y)
            node_data.append({'pos': (float(cx), float(cy)), 'area': float(area)})

    node_data.sort(key=lambda d: (d['pos'][1], d['pos'][0]))

    # Build pattern with nodes
    pat = Pattern(name=name)
    for k, data in enumerate(node_data, start=1):
        pat.add_node(f"N{k}", data['pos'], size=data['area'])

    # Helper: sample along a segment to check if it's painted green
    def link_exists(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L > max_pair_px: 
            return False
        n = max(2, int(L))  # 1px step
        xs = np.linspace(x1, x2, n).astype(np.int32)
        ys = np.linspace(y1, y2, n).astype(np.int32)
        xs = np.clip(xs, 0, green.shape[1]-1)
        ys = np.clip(ys, 0, green.shape[0]-1)
        line_vals = green[ys, xs] > 0
        hit_rate = line_vals.mean()
        # also ensure there isn't a long break
        gaps = 0
        max_gap_seen = 0
        for v in line_vals:
            if v: gaps = 0
            else:
                gaps += 1
                max_gap_seen = max(max_gap_seen, gaps)
        return (hit_rate >= link_hit_thresh) and (max_gap_seen <= max_link_gap)

    # Create edges by probing pairs (O(N^2) is fine for small constellations)
    labels = list(pat.nodes.keys())
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a, b = labels[i], labels[j]
            p1 = pat.nodes[a].position
            p2 = pat.nodes[b].position
            if link_exists(p1, p2):
                pat.add_edge(a, b)

    return pat, white, green
