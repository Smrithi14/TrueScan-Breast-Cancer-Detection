import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features_gray(img_gray, distances=[1], angles=[0], levels=256):
    glcm = graycomatrix(img_gray, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    return [
        graycoprops(glcm, 'contrast')[0,0],
        graycoprops(glcm, 'dissimilarity')[0,0],
        graycoprops(glcm, 'homogeneity')[0,0],
        graycoprops(glcm, 'energy')[0,0],
        graycoprops(glcm, 'correlation')[0,0]
    ]

def extract_features_from_path(img_path, size=(50,50)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return extract_glcm_features_gray(img)
