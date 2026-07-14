import numpy as np
import cv2

from canon.T2.process import  epipolar_geometry
from canon.T2.plotting import visualization


def _filter_points_by_reprojection(points_3d, image_points, R, t, K, reproj_thresh=3.0):
    """
    Remove pontos 3D cujo erro de reprojeção é maior que o limite.
    
    Args:
        points_3d: (N,3) pontos reconstruídos.
        image_points: (N,2) pontos 2D correspondentes.
        R, t: pose da câmera.
        K: matriz intrínseca da câmera.
        reproj_thresh: limite em pixels para considerar inlier.
    
    Returns:
        inliers_3d: pontos 3D filtrados.
        mask: array booleano indicando quais pontos foram mantidos.
    """
    proj_points, _ = cv2.projectPoints(points_3d, cv2.Rodrigues(R)[0], t, K, None)
    proj_points = proj_points.reshape(-1, 2)
    
    errors = np.linalg.norm(proj_points - image_points, axis=1)
    mask = errors < reproj_thresh
    return points_3d[mask], mask


def _register_new_image(img, features, match_results, cameras, used_images, global_points, K, triangulator, image_points_dict):
    """
    Registra uma nova imagem no SfM incremental:
    - Busca correspondências 2D–3D
    - Estima pose com solvePnPRansac
    - Triangula novos pontos com uma imagem de referência
    - Atualiza a nuvem global
    
    Retorna True se a imagem foi registrada com sucesso.
    """
    print(f"\n>> Registrando imagem {img}")

    # Procurar matches com imagens já registradas
    all_matches = []
    for ref_img in used_images:
        key = (ref_img, img) if (ref_img, img) in match_results else (img, ref_img)
        if key in match_results:
            all_matches.append((ref_img, match_results[key], key))
    
    if not all_matches:
        print("Sem matches suficientes, pulando...")
        return False

    # Escolher referência com mais matches
    ref_img, match_data, key = max(all_matches, key=lambda x: x[1]['num_matches'])
    kp_ref, _ = features[ref_img]
    kp_new, _ = features[img]

    pts_ref = match_data['pts1'] if ref_img == key[0] else match_data['pts2']
    pts_new = match_data['pts2'] if ref_img == key[0] else match_data['pts1']

    object_points = np.array(global_points[:len(pts_new)], dtype=np.float32) # fake associações 
    image_points = pts_new[:len(object_points)].reshape(-1,2)

    # object_points = np.array(object_points, dtype=np.float32)
    # image_points = np.array(image_points, dtype=np.float32)

    if len(object_points) < 6:
        print("Poucos pontos 2D-3D, pulando...")
        return False

    # --- Estimar pose com PnP robusto ---
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, None
    )
    if not success:
        print("PnP falhou, pulando...")
        return False

    R_new, _ = cv2.Rodrigues(rvec)
    t_new = tvec
    cameras[img] = (R_new, t_new)
    used_images.add(img)

    # Guardar correspondências 2D-3D para reprojeção
    image_points_dict[img] = image_points

    # --- Triangular novos pontos com a referência ---
    pts_4d = triangulator.triangulate_points(
        pts_ref, pts_new,
        cameras[ref_img][0], cameras[ref_img][1],
        R_new, t_new
    )
    new_points_3d = triangulator.convert_to_3d(pts_4d)

    # Filtrar por reprojeção
    new_points_3d = np.array(new_points_3d, dtype=np.float32)
    image_points_new = pts_new.reshape(-1, 2)[:len(new_points_3d)]

    inliers_3d, mask = _filter_points_by_reprojection(
        new_points_3d, image_points_new, R_new, t_new, K, reproj_thresh=5.0
    )

    print(f"Triangulados {len(new_points_3d)} pontos, mantidos {len(inliers_3d)} inliers")

    global_points.extend(inliers_3d.tolist())
    image_points_dict[img] = image_points_new[mask]   # armazenar apenas inliers

    return True


def construct_point_cloud(features, match_results, K):
    # Cria estimador
    epipolar_estimator = epipolar_geometry.EpipolarGeometryEstimator(
        camera_matrix=K,  # Começa sem calibração
        ransac_threshold=1.0,
        confidence=0.99
    )
        
    triangulator = epipolar_geometry.Triangulator(K)
    
    best_pair = max(match_results.items(), key=lambda x: x[1]['num_matches'])
    (img1, img2), match_data = best_pair
    
    E, mask = epipolar_estimator.estimate_essential_matrix(pts1, pts2)
    _, R, t, tri_mask = epipolar_estimator.recover_pose(E, pts1, pts2, mask)
    
    pts1, pts2 = match_data['pts1'], match_data['pts2']
    # Inicializar câmeras
    cameras = {
        img1: (np.eye(3), np.zeros((3,1))),  # identidade
        img2: (R, t)
    }

    # Triangular pontos iniciais
    pts_4d = triangulator.triangulate_points(pts1, pts2,
                                                cameras[img1][0], cameras[img1][1],
                                                cameras[img2][0], cameras[img2][1])
    points_3d = triangulator.convert_to_3d(pts_4d).tolist()

    # Criar dicionário de correspondências 2D-3D
    image_points_dict = {}
    image_points_dict[img1] = pts1.reshape(-1, 2)[:len(points_3d)]
    image_points_dict[img2] = pts2.reshape(-1, 2)[:len(points_3d)]

    # --- Passo 2: adicionar novas imagens ---
    used_images = {img1, img2}
    remaining_images = set(features.keys()) - used_images

    for img in remaining_images:
        success = _register_new_image(
            img=img,
            features=features,
            match_results=match_results,
            cameras=cameras,
            used_images=used_images,
            global_points=points_3d,
            K=K,
            triangulator=triangulator,
            image_points_dict=image_points_dict
        )
        if success:
            print(f"Imagem {img} registrada com sucesso!")
        else:
            print(f"Imagem {img} não pôde ser registrada.")
            
            
    return points_3d, cameras, image_points_dict