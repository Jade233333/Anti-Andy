import cv2
import numpy as np
from scipy.ndimage import rotate

def auto_crop(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Binary thresholding and inversion
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Get bounding box coordinates from the largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Crop the image
    cropped = image[y:y+h, x:x+w]

    return cropped


def determine_score(arr, angle):
    data = rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
    return histogram, score

def correct_skew(image, delta=1, limit=10):
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to convert the image to black and white
    _, black_white = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Reduce noise using a median filter
    denoised = cv2.medianBlur(black_white, 1)

    # Correct skew
    corrected = correct_skew(denoised)

    # Auto crop
    cropped = auto_crop(corrected)

    inverted = cv2.bitwise_not(cropped)

    return inverted


# Example usage
image_path = 'test/1.png'
preprocessed_image = preprocess_image(image_path)
cv2.imwrite("processed_image.png", preprocessed_image)

