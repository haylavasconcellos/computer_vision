import networkx as nx
import cv2
from canon.T1.process import feature_extraction
from canon.T1.plotting import plotting
import numpy as np
import itertools

def hamiltonian_path_brute_force(G: nx.Graph) -> list[int]:
    n = G.number_of_nodes()
    best_path: list[int] | None = None
    
    max_cost = 0

    # Try all permutations of nodes
    for perm in itertools.permutations(G.nodes):
        path = list(perm)
        valid = True
        cost = 0

        # Check if consecutive nodes are connected
        for i in range(n - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                cost += G[u][v].get('weight', 1)  # default weight=1 if not set
            else:
                valid = False
                break

        # Update best path if valid and cheaper
        if valid and cost > max_cost:
            max_cost = cost
            best_path = path

    return best_path


def hamiltonian_path_heuristic(G: nx.Graph) -> list[int]:
    n = G.number_of_nodes()
    
    # começa com a aresta de maior peso
    u, v, w = max(G.edges(data="weight"), key=lambda x: x[2])
    path = [u, v]  # inicia o caminho com os dois vértices da aresta máxima
    
    # enquanto não tiver todos os nós
    while len(path) < n:
        candidates = []
        
        # tenta expandir pela ponta esquerda
        for neighbor, data in G[path[0]].items():
            if neighbor not in path:
                candidates.append((data["weight"], neighbor, "left"))
        
        # tenta expandir pela ponta direita
        for neighbor, data in G[path[-1]].items():
            if neighbor not in path:
                candidates.append((data["weight"], neighbor, "right"))
        
        if not candidates:
            break  # não conseguiu expandir mais
        
        # pega a aresta candidata de maior custo
        w, node, side = max(candidates, key=lambda x: x[0])
        
        if side == "left":
            path.insert(0, node)
        else:
            path.append(node)
    
    return path


def build_match_graph(kp_descs: list[np.ndarray], n: int) -> tuple[nx.Graph, dict[int, list[cv2.DMatch]]]:
    match_graph = {x: [] for x in range(n)}
    
    matches_per_image = {x: [] for x in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            bf = cv2.BFMatcher(cv2.NORM_L2)

            kp1, des1 = kp_descs[i]
            kp2, des2 = kp_descs[j]

            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = plotting.david_lowe_ratio_test(matches)
            
            if len(good_matches) < 4:
                continue
                    
            match_graph[i].append([j, len(good_matches)])
            
            matches_per_image[i].extend(good_matches)
            
    G = nx.Graph()

    # Add edges with weights
    for node, neighbors in match_graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)
            
    return G, matches_per_image


def is_inverted(path: list[int], 
                images: list[np.ndarray], 
                matches_per_image: dict[int, list[cv2.DMatch]], 
                kp_descs: tuple[list[cv2.KeyPoint], np.ndarray]):
    
    """Calcula o quão a esquerda estão os keypoints que deram match de cada uma das pontas do caminho
    para saber se está invertido"""
    
    begin_path = path[0]

    keypoints_match_begin = []
    for match in matches_per_image[begin_path]:
        keypoints_match_begin.append(kp_descs[begin_path][0][match.queryIdx])
        
    end_path = path[-1]

    keypoints_match_end = []
    for match in matches_per_image[end_path]:
        keypoints_match_end.append(kp_descs[end_path][0][match.queryIdx])
    
    width_image_begin = images[begin_path].shape[1]

    left_score_begin = 0

    for keypoint in keypoints_match_begin:
        left_score_begin += 1 - (keypoint.pt[0] / width_image_begin) # y coordinate
    
    
    width_image_end = images[end_path].shape[1]

    left_score_end = 0

    for keypoint in keypoints_match_end:
        left_score_end += 1 - (keypoint.pt[0] / width_image_end) # y coordinate
        
    return left_score_begin > left_score_end
    


def find_order(images: list[np.ndarray]) -> list[int]:
    kp_descs = [feature_extraction.SIFT(img, nfeatures=1000) for img in images]
    
    G, matches_per_image = build_match_graph(kp_descs, len(images))
    
    if len(images) < 10:
        path = hamiltonian_path_brute_force(G)
    else:
        path = hamiltonian_path_heuristic(G)
        
    # path can be inverted
    if (is_inverted(path, images, matches_per_image, kp_descs)):
        return path[::-1]
    
    return path
