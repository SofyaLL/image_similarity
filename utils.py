from typing import List, Tuple

import cv2
import numpy as np

orb = cv2.ORB_create(nfeatures=150)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def compute_orb(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Compute ORB (Oriented FAST and Rotated BRIEF) keypoints and descriptors for a given image.

    Parameters:
    image (numpy.ndarray): The input grayscale image for which to compute the ORB keypoints and descriptors.

    Returns:
    - key_points (List[cv2.KeyPoint]): A list of detected keypoints.
    - descriptors (np.ndarray): An array of computed descriptors corresponding to the keypoints.
    """
    key_points, descriptors = orb.detectAndCompute(image, None)
    return key_points, descriptors


def calculate_matches(descriptor_1: np.ndarray, descriptor_2: np.ndarray) -> List[List[cv2.DMatch]]:
    """
    Calculate good matches between two sets of ORB descriptors using the BFMatcher with the KNN algorithm.

    Parameters:
    descriptor_1 (np.ndarray): Descriptors of the first image.
    descriptor_2 (np.ndarray): Descriptors of the second image.

    Returns:
    List[List[cv2.DMatch]]:
        A list of good matches. Each good match is a list containing a single DMatch object.
    """
    ratio = 0.89
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)
    good_matches_1 = [[first] for first, second in matches if first.distance < second.distance * ratio]

    matches_2 = bf.knnMatch(descriptor_2, descriptor_1, k=2)
    good_matches_2 = [[first] for first, second in matches_2 if first.distance < second.distance * ratio]

    good_matches = [
        match1
        for match2 in good_matches_2
        for match1 in good_matches_1
        if match1[0].queryIdx == match2[0].trainIdx and match1[0].trainIdx == match2[0].queryIdx
    ]
    return good_matches


def image_resize(image: np.ndarray, maxD: int = 256) -> np.ndarray:
    """
    Resize an image while maintaining its aspect ratio, with the maximum dimension set to maxD pixels.

    Parameters:
    image (np.ndarray): The input image to be resized. This can be a 2D grayscale or a 3D color image.
    maxD (int): The maximum dimension (width or height) to resize the image to. Default is 256 pixels.

    Returns:
    np.ndarray: The resized image with its aspect ratio preserved.
    """
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width, _ = image.shape
    aspect_ratio = width / height
    if aspect_ratio < 1:
        new_size = (int(maxD * aspect_ratio), maxD)
    else:
        new_size = (maxD, int(maxD / aspect_ratio))
    image = cv2.resize(image, new_size)
    return image


def calculate_score(
    keypoints_1: List[cv2.KeyPoint],
    descriptor_1: np.ndarray,
    keypoints_2: List[cv2.KeyPoint],
    descriptor_2: np.ndarray,
) -> float:
    """
    Calculate the matching score between two sets of ORB keypoints and descriptors.

    Parameters:
    keypoints_1 (list): A list of keypoints from the first image.
    descriptor_1 (numpy.ndarray): Descriptors corresponding to the keypoints of the first image.
    keypoints_2 (list): A list of keypoints from the second image.
    descriptor_2 (numpy.ndarray): Descriptors corresponding to the keypoints of the second image.

    Returns:
    float: A matching score between 0 and 100, rounded to three decimal places.
           The score is calculated as the ratio of the number of good matches
           to the maximum number of keypoints in either image expressed as a percentage.
    """
    matches = calculate_matches(descriptor_1, descriptor_2)
    score = 100 * (len(matches) / max(len(keypoints_1), len(keypoints_2)))
    return round(score, 3)
