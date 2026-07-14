from canon import config
import argparse
from pathlib import Path
import logging
import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import copy
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from canon.T2.utils import metrics
from canon.utils import image_utils
from canon.T2.process import feature_extraction, epipolar_geometry, reconstruction_3d
from canon.T2.plotting import visualization

DATASET = "GustavII"
DESCRIPTOR = 'sift'
LOWE = 0.7
RANSAC_TH = 5
RANSAC_PROB = 0.80

OUTNAME = f"{DATASET}_{DESCRIPTOR}_rsacProb{RANSAC_PROB}_{RANSAC_TH}_lowe_{LOWE}"

def findFeaturesFilter(img1, img2):
    if(DESCRIPTOR == 'sift'):
        pts0, pts1 = feature_extraction.find_features(img1, img2, LOWE)
    elif(DESCRIPTOR == 'akaze'):
        pts0, pts1 =  feature_extraction.find_features_akaze(img1, img2, LOWE)
    elif(DESCRIPTOR == 'orb'):
        pts0, pts1 =  feature_extraction.find_features_orb(img1, img2, LOWE)

    return pts0, pts1


def build_3d_image(img_dir, res_dir, densify = False):
    # Input Camera Intrinsic Parameters
    K = epipolar_geometry.get_intrinsic_matrix()
    # img_list = sorted(os.listdir(img_dir))
    # K = epipolar_geometry.estimate_epipolar_matrix(cv2.imread(f'{img_dir}/1.jpg'))

    downscale = 2
    K[0,0] = K[0,0] / float(downscale)
    K[1,1] = K[1,1] / float(downscale)
    K[0,2] = K[0,2] / float(downscale)
    K[1,2] = K[1,2] / float(downscale)

    # Suppose if computationally heavy, then the images can be downsampled once. Note that downsampling is done in powers of two, that is, 1,2,4,8,...

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    posearr = K.ravel()
    R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    R_t_1 = np.empty((3, 4))

    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3, 4))

    Xtot = np.zeros((1, 3))
    colorstot = np.zeros((1, 3))
    
    img_list = sorted(os.listdir(img_dir))
    images = []
    for img in img_list:
        if '.jpg' in img.lower() or '.png' in img.lower():
            images = images + [img]
    i = 0
    
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

      # Added in case we will merge densification step in this code. Mostly it will be considered separately, though still added here.

    # Setting the Reference two frames
    img0 = image_utils.img_downscale(cv2.imread(img_dir + '/' + images[i]), downscale)
    img1 = image_utils.img_downscale(cv2.imread(img_dir + '/' + images[i + 1]), downscale)

    pts0, pts1 = findFeaturesFilter(img0, img1)

    # Finding essential matrix
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=RANSAC_PROB, threshold=RANSAC_TH, mask=None)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    # The pose obtained is for second image with respect to first image
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)  # |finding the pose
    pts0 = pts0[mask.ravel() > 0]
    pts1 = pts1[mask.ravel() > 0]
    R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
    R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())

    P2 = np.matmul(K, R_t_1)

    # Triangulation is done for the first image pair. The poses will be set as reference, that will be used for increemental SfM
    pts0, pts1, points_3d = epipolar_geometry.Triangulation(P1, P2, pts0, pts1, K, repeat=False)
    # Backtracking the 3D points onto the image and calculating the reprojection error. Ideally it should be less than one.
    # If found to be the culprit for an incorrect point cloud, enable Bundle Adjustment
    error, points_3d, repro_pts = epipolar_geometry.ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 1)
    print("REPROJECTION ERROR: ", error)
    Rot, trans, pts1, points_3d, pts0t = epipolar_geometry.PnP(points_3d, pts1, K, np.zeros((5, 1), dtype=np.float32), pts0, initial=1)
    #Xtot = np.vstack((Xtot, points_3d))

    R = np.eye(3)
    t = np.array([[0], [0], [0]], dtype=np.float32)

    # Here, the total images to be take into consideration can be varied. Ideally, the whole set can be used, or a part of it. For whole lot: use tot_imgs = len(images) - 2
    tot_imgs = len(images) - 2 

    posearr = np.hstack((posearr, P1.ravel()))
    posearr = np.hstack((posearr, P2.ravel()))

    gtol_thresh = 0.5
    #camera_orientation(path, mesh, R_t_0, 0)
    #camera_orientation(path, mesh, R_t_1, 1)

    reprojError = []
    for i in tqdm(range(tot_imgs)):
        # Acquire new image to be added to the pipeline and acquire matches with image pair
        img2 = image_utils.img_downscale(cv2.imread(img_dir + '/' + images[i + 2]), downscale)

        # pts0,pts1 = find_features(img1,img2)

        pts_, pts2 = findFeaturesFilter(img1, img2)
        if i != 0:
            pts0, pts1, points_3d = epipolar_geometry.Triangulation(P1, P2, pts0, pts1, K, repeat = False)
            pts1 = pts1.T
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
            points_3d = points_3d[:, 0, :]
        
        # There gone be some common point in pts1 and pts_
        # we need to find the indx1 of pts1 match with indx2 in pts_
        indx1, indx2, temp1, temp2 = feature_extraction.common_points(pts1, pts_, pts2)
        com_pts2 = pts2[indx2]
        com_pts_ = pts_[indx2]
        com_pts0 = pts0.T[indx1]
        # We have the 3D - 2D Correspondence for new image as well as point cloud obtained from before. The common points can be used to find the world coordinates of the new image
        # using Perspective - n - Point (PnP)
        print("3D points:", points_3d[indx1].shape)
        print("2D points:", com_pts2.shape)
        
        Rot, trans, com_pts2, points_3d, com_pts_ = epipolar_geometry.PnP(points_3d[indx1], com_pts2, K, np.zeros((5, 1), dtype=np.float32), com_pts_, initial = 0)
        # Find the equivalent projection matrix for new image
        Rtnew = np.hstack((Rot, trans))
        Pnew = np.matmul(K, Rtnew)

        #print(Rtnew)
        error, points_3d, _ = epipolar_geometry.ReprojectionError(points_3d, com_pts2, Rtnew, K, homogenity = 0)
    
        
        temp1, temp2, points_3d = epipolar_geometry.Triangulation(P2, Pnew, temp1, temp2, K, repeat = False)
        error, points_3d, _ = epipolar_geometry.ReprojectionError(points_3d, temp2, Rtnew, K, homogenity = 1)
        reprojError.append(error)
        print("Reprojection Error: ", error)
        # We are storing the pose for each image. This will be very useful during multiview stereo as this should be known
        posearr = np.hstack((posearr, Pnew.ravel()))

        Xtot = np.vstack((Xtot, points_3d[:, 0, :]))
        pts1_reg = np.array(temp2, dtype=np.int32)
        colors = np.array([img2[l[1], l[0]] for l in pts1_reg.T])
        colorstot = np.vstack((colorstot, colors)) 


        R_t_0 = np.copy(R_t_1)
        P1 = np.copy(P2)
        plt.scatter(i, error)
        plt.title("Reprojection Error")
        plt.pause(0.05)

        img0 = np.copy(img1)
        img1 = np.copy(img2)
        pts0 = np.copy(pts_)
        pts1 = np.copy(pts2)
        #P1 = np.copy(P2)
        P2 = np.copy(Pnew)
        cv2.imshow('image', img2)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    #plt.show()
    plt.title(f"{DESCRIPTOR} - RANSAC TH {RANSAC_TH}: Erro Reprojeção")
    plt.savefig(f"{OUTNAME}.png", dpi=300, bbox_inches="tight")

    cv2.destroyAllWindows()

    # Finally, the obtained points cloud is registered and saved using open3d. It is saved in .ply form, which can be viewed using meshlab
    print("Processing Point Cloud...")
    print(Xtot.shape, colorstot.shape)
    image_utils.to_ply(res_dir, Xtot, colorstot, densify, filename=f"{OUTNAME}")
    print("Done!")

    # --- Metrics reporting ---
    print("\n=== Reconstruction Metrics ===")
    num_sparse_points = Xtot.shape[0]
    num_dense_points = 0 if not densify else Xtot.shape[0]  # placeholder until MVS step
    print(f"Sparse points: {num_sparse_points}")
    print(f"Dense points: {num_dense_points}")

    # Compute bounding box + density
    bbox_stats = metrics.compute_bounding_box(Xtot)
    print(f"Bounding box dimensions (m): "
          f"dx={bbox_stats['dx']:.2f}, "
          f"dy={bbox_stats['dy']:.2f}, "
          f"dz={bbox_stats['dz']:.2f}")
    print(f"Density (points/m²): {bbox_stats['density']:.2f}")

    reprojError = np.array(reprojError)
    # Compute reprojection error of last registration
    reproj_stats = metrics.compute_reprojection_errors(points_3d, com_pts2, Rtnew, K, homogenity=0)
    print(f"Reprojection error (mean): {reproj_stats['mean_error']:.2f}")
    print(f"Reprojection error OK(mean): {reprojError.mean():.2f}")
    print(f"Reprojection error (median): {reproj_stats['median_error']:.2f}")
    print(f"Reprojection error OK (median): {np.median(reprojError):.2f}")

    with open(f"{res_dir}/{OUTNAME}.txt", "w") as f:
        f.write(f"Sparse points: {num_sparse_points}\n")
        f.write(f"Dense points: {num_dense_points}\n")

        f.write(f"Bounding box dimensions (m): "
          f"dx={bbox_stats['dx']:.2f}, "
          f"dy={bbox_stats['dy']:.2f}, "
          f"dz={bbox_stats['dz']:.2f}\n")
        f.write(f"Density (points/m²): {bbox_stats['density']}\n\n")
        
        f.write(f"Reprojection error (mean): {reprojError.mean():.2f}\n")
        f.write(f"Reprojection error (median): {np.median(reprojError):.2f}")



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # Configurar logger
    logger = config.setup_logger("3D main pipeline")
    
    # Configurar argparse para receber argumentos
    parser = argparse.ArgumentParser(description="Pipeline de reconstrução 3D")
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Caminho para a pasta com as imagens",
        required=True
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Caminho para salvar resultado",
        required=True
    )
    parser.add_argument(
        "--densify",
        type=str2bool,
        required=True,
        help="Produzir densificação (True/False)"
    )
    
    args = parser.parse_args()

    image_dir = args.image_dir
    res_dir = args.res_dir
    densify = args.densify

    # Executar pipeline    
    build_3d_image(image_dir, res_dir, densify)

    # Visualizar nuvem de pontos
    point_cloud_file = os.path.join(res_dir, "dense.ply" if densify else f"{OUTNAME}.ply")
    visualization.visualize_point_cloud(point_cloud_file)

#python src/canon/T2/main.py --image_dir data/T2/interim/GustavIIAdolf --res_dir data/T2/interim/GustavIIAdolf/mainRun --densify False
