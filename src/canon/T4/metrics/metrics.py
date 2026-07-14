import torch
import lpips
import numpy as np
import torch.nn as nn
import cv2
from torchvision import transforms
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from basicsr.metrics.niqe import calculate_niqe
from torchvision.models import inception_v3
from scipy import linalg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
Compare 2 images using LPIPS score.

Args:
    img1: Input image as a NumPy array.
    img2: "
    net: Type of nn in which LPIPS will run. ['alex','vgg','squeeze']

Returns:
    LPIPS score. Value in (0, 1)
"""
def LPIPS(img1 : np.array,
          img2 : np.array,
          net : str = "alex"):
    fn_perda = lpips.LPIPS(net).to(DEVICE) #LPIPS pre treinado

    preproc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -> Normaliza para intervalo (-1, 1) que o LPIPS precisa
        transforms.Resize((256, 256)),
    ])

    img1_tensor = preproc(img1).unsqueeze(0) # adiciona dimensao batch
    img2_tensor = preproc(img2).unsqueeze(0)

    d = fn_perda(img1_tensor.to(DEVICE), img2_tensor.to(DEVICE))

    return d.item()

"""
Compare 2 images using Structural Similarity score.

Args:
    img1: Input image as a NumPy array.
    img2: "
    channel_axis: index of image channel dim. (None if grayscale)

Returns:
    A tuple of (score, diff).
    - keypoints: SSIM score. Value in (0, 1)
    - diff: An image representation of the structural similarity
"""
def SSIM(img1 : np.array, img2 : np.array, channel_axis : int = 2): 
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    (score, diff) = ssim(img1, img2, full=True, channel_axis=channel_axis)
    diff = (diff * 255).astype("uint8")

    return score, diff


"""
    NIQE is a no-reference image quality metric. It compares the
    statistical distribution of the input image's MSCN coefficients
    against a model derived from natural scene statistics (NSS).
    The calculation does not require a reference image.

Args:
    img1: Input image as a NumPy array.
    crop_border: how many pixels to crop on the image border

Returns:
    - NIQE Score, sub 10 is very good. The bigger = The worse 
"""
def NIQE(img : np.array, crop_border: int = 0):
    return calculate_niqe(img, crop_border=crop_border)

#===================================================================================
# FID (Frechet)


"""
    FID is a real/generated image set comparator. 

Args:
    realSet: list containing real images.
    gennSet: " generated ".

Returns:
    - FID Score. The bigger = The worse 
"""
def FID(realSet : list, genSet : list):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize((299, 299))
        ])

    realSet = [preprocess(np.array(x)).unsqueeze(0) for x in realSet]
    genSet = [preprocess(np.array(x)).unsqueeze(0) for x in genSet]

    realSet = torch.cat(realSet, dim=0).to(DEVICE)  
    genSet = torch.cat(genSet, dim=0).to(DEVICE)  

    model = __InceptionV3Features().to(DEVICE).eval()

    with torch.no_grad():
        fv1 = model(realSet).cpu().numpy()
        fv2 = model(genSet).cpu().numpy()

    return __FID_compute(fv1, fv2)


def __FID_compute(fv1, fv2):
    # fv1, fv2 must be shape (N, 2048)
    mu1 = np.mean(fv1, axis=0)
    mu2 = np.mean(fv2, axis=0)

    sigma1 = np.cov(fv1, rowvar=False)
    sigma2 = np.cov(fv2, rowvar=False)

    if sigma1.ndim == 0: 
        #hmmmm
        sigma1 = np.zeros((fv1.shape[1], fv1.shape[1]), dtype=np.float64)
    if sigma2.ndim == 0:
        sigma2 = np.zeros((fv2.shape[1], fv2.shape[1]), dtype=np.float64)

    diff = mu1 - mu2
    diff_squared = diff.dot(diff)

    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fid)


class __InceptionV3Features(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        inception.fc = nn.Identity()      # remove classifier
        inception.dropout = nn.Identity() # remove dropout
        # tira camada final, queremos o feature vector de 2048 valores 
        self.model = inception

    def forward(self, x):
        x = self.model(x)     # produces (N, 2048)
        return x  # shape: (N, 2048)