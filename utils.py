import heapq
import pickle
import time
import traceback

import cv2

orb = cv2.ORB_create(nfeatures=150)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def load_data_from_pickle(pickle_path):
    with open(pickle_path, "rb") as file:
        try:
            data = pickle.load(file)
            return data
        except:
            print(traceback.format_exc())


def save_data_to_pickle(pickle_path, data_base):
    with open(pickle_path, "wb") as file:
        pickle.dump(data_base, file)


def compute_orb(image):
    """
    Compute ORB (Oriented FAST and Rotated BRIEF) keypoints and descriptors for a given image.

    Parameters:
    image (numpy.ndarray): The input image for which to compute the ORB keypoints and descriptors.
                           This should be a grayscale image of type numpy array.

    Returns:
    tuple: A tuple containing:
        - key_points (list): A list of detected keypoints. Each keypoint is an instance of cv2.KeyPoint.
        - descriptors (numpy.ndarray): An array of computed descriptors corresponding to the keypoints.
                                       Each descriptor is a vector of integers.
    """
    key_points, descriptors = orb.detectAndCompute(image, None)
    return (key_points, descriptors)


def calculate_matches(descriptor_1, descriptor_2):
    """
    Calculate good matches between two sets of ORB descriptors using the BFMatcher with the KNN algorithm.

    Parameters:
    descriptor_1 (numpy.ndarray): Descriptors of the first image.
    descriptor_2 (numpy.ndarray): Descriptors of the second image.

    Returns:
    list: A list of good matches. Each good match is a list containing a single DMatch object,
          where a DMatch object has attributes queryIdx, trainIdx, imgIdx, and distance.
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


def image_resize(image):
    """
    Resize a grayscale image while maintaining its aspect ratio, with the maximum dimension set to 256 pixels.

    Parameters:
    image (numpy.ndarray): The input grayscale image to be resized. This should be a 2D numpy array with shape (height, width).

    Returns:
    numpy.ndarray: The resized image with its aspect ratio preserved, and the maximum dimension set to 256 pixels.
    """
    maxD = 256
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


def find_top_images(target_path, database, num_top_images=4):
    target_image = image_resize(cv2.imread(target_path))
    start_time = time.perf_counter()
    target_keypoints, target_descriptor = compute_orb(target_image)
    top_image_ids = []
    heapq.heapify(top_image_ids)

    for image_id in database:
        image_descriptor = database[image_id]["descriptor"]
        image_num_keypoints = database[image_id]["n_keypoints"]
        try:
            current_matches = calculate_matches(target_descriptor, image_descriptor)
            current_score = 100 * (len(current_matches) / max(len(target_keypoints), image_num_keypoints))
        except:
            current_score = 0

        heapq.heappush(top_image_ids, (current_score, image_id))
        if len(top_image_ids) > num_top_images:
            heapq.heappop(top_image_ids)

    result = [
        {"top_image_id": image, "score": round(score, 3)}
        for score, image in sorted(top_image_ids, reverse=True)
    ]
    search_time = time.perf_counter() - start_time
    return result, round(search_time, 3)


def calculate_score(keypoints_1, descriptor_1, keypoints_2, descriptor_2):
    """
    Calculate the matching score between two sets of ORB keypoints and descriptors.

    Parameters:
    keypoints_1 (list): A list of keypoints from the first image. Each keypoint is an instance of cv2.KeyPoint.
    descriptor_1 (numpy.ndarray): Descriptors corresponding to the keypoints of the first image. Each descriptor is a vector of integers.
    keypoints_2 (list): A list of keypoints from the second image. Each keypoint is an instance of cv2.KeyPoint.
    descriptor_2 (numpy.ndarray): Descriptors corresponding to the keypoints of the second image. Each descriptor is a vector of integers.

    Returns:
    float: A matching score between 0 and 100, rounded to three decimal places.
           The score is calculated as the ratio of the number of good matches
           to the maximum number of keypoints in either image expressed as a percentage.
    """
    matches = calculate_matches(descriptor_1, descriptor_2)
    score = 100 * (len(matches) / max(len(keypoints_1), len(keypoints_2)))
    return round(score, 3)
