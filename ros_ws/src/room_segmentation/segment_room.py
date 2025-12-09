import cv2
import numpy as np
from random import randint
from typing import Optional, Tuple, List, Union
from skimage.transform import resize


def segment_rooms(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    noise_threshold: int = 25,
    apply_mask: bool = True,
    dist_threshold_min: float = 0.3,
    dist_threshold_max: float = 1.0,
    resize_dim: int = 500,
    min_area: int = -1,
) -> np.ndarray:
    """
    使用分水岭算法进行房间分割。通过距离变换和形态学操作，
    尝试识别图像中的房间区域。
    """
    img = image.copy()

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_height, original_width = img.shape
    img = cv2.resize(img, (resize_dim, int(
        resize_dim * img.shape[0] / img.shape[1])))

    blurred_img = cv2.GaussianBlur(
        img, (blur_kernel_size, blur_kernel_size), 0)

    _, binary_img = cv2.threshold(
        blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if apply_mask:
        mask = np.zeros_like(img)
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > noise_threshold:
                cv2.fillPoly(mask, [contour], 255)
        img[mask == 0] = 0

    laplacian_kernel = np.array(
        [[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    laplacian = cv2.filter2D(img, cv2.CV_32F, laplacian_kernel)
    img = np.float32(img) - laplacian

    img = np.clip(img, 0, 255).astype("uint8")

    _, binary_threshold = cv2.threshold(
        img, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    distance_transform = cv2.distanceTransform(
        binary_threshold, cv2.DIST_L2, 3)
    cv2.normalize(distance_transform, distance_transform,
                  0, 1.0, cv2.NORM_MINMAX)

    _, dist_thresholded = cv2.threshold(
        distance_transform, dist_threshold_min, dist_threshold_max, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(dist_thresholded, kernel)

    contours, _ = cv2.findContours(dilated.astype(
        "uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = np.zeros_like(dilated, dtype=np.int32)

    for idx, contour in enumerate(contours):
        cv2.drawContours(markers, contours, idx, color=(idx + 1), thickness=-1)

    cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), markers)

    markers = resize(markers, (original_height, original_width),
                     order=0, preserve_range=True, anti_aliasing=False)

    if min_area > 0:
        markers = filter_small_zones(markers, min_area=min_area)

    return markers


def visualize_segments(
    markers: np.ndarray, colors: Optional[List[Tuple[int]]] = None, show_labels: bool = False
) -> np.ndarray:
    """
    根据分割标记生成彩色图像。可以提供颜色列表，如果设置了show_labels，则在每个区域中心标记区域索引。
    """
    colored_img = np.zeros_like(markers, dtype=np.uint8)
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_GRAY2BGR)

    unique_markers = np.unique(markers)
    unique_markers = unique_markers[(
        unique_markers > 0) & (unique_markers < 255)]

    if colors is None:
        colors = {index: (randint(0, 255), randint(0, 255),
                          randint(0, 255)) for index in unique_markers}

    for index in unique_markers:
        colored_img[markers == index] = colors[index]

    if show_labels:
        for index in unique_markers:
            y_coords, x_coords = np.where(markers == index)
            y_center = int(np.mean(y_coords))
            x_center = int(np.mean(x_coords))
            cv2.putText(
                colored_img,
                str(index + 1),
                (x_center, y_center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return colored_img


def filter_small_zones(markers: np.ndarray, min_area: float = 1500.0) -> np.ndarray:
    """
    对分割标记进行阈值处理，消除面积小于指定阈值的区域，并将其设置为周围最常见的区域标签。
    """
    markers = markers.copy()
    unique_markers = np.unique(markers)
    unique_markers = unique_markers[(
        unique_markers > 0) & (unique_markers < 255)]

    for index in unique_markers:
        mask = (markers == index)
        y_coords, x_coords = np.where(mask)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        area = (y_max - y_min) * (x_max - x_min)
        if area < min_area:
            y_min = max(int(y_min - (y_max - y_min) * 0.1), 0)
            y_max = min(int(y_max + (y_max - y_min) * 0.1), markers.shape[0])
            x_min = max(int(x_min - (x_max - x_min) * 0.1), 0)
            x_max = min(int(x_max + (x_max - x_min) * 0.1), markers.shape[1])

            surrounding = markers[y_min:y_max, x_min:x_max]
            surrounding = surrounding[(surrounding != index) & (
                surrounding > 0) & (surrounding < 255)]

            if len(surrounding) == 0:
                continue

            most_common = np.bincount(surrounding).argmax()
            markers[mask] = most_common

    return markers


def identify_zones(
    img: np.ndarray,
    unknown_label: Union[int, Tuple[int, int, int]],
    empty_label: Union[int, Tuple[int, int, int]],
):
    """
    接受图像并尝试将其分割为已知、未知和其他（基本上是墙壁）区域。
    """
    base = np.ones_like(img)
    base = np.where(img == unknown_label, -1, base)
    base = np.where(img == empty_label, 0, base)
    return base


if __name__ == "__main__":
    img = cv2.imread("./house.jpeg", cv2.IMREAD_GRAYSCALE)
    segmented_markers = segment_rooms(img)
    filtered_markers = filter_small_zones(segmented_markers)
    final_image = visualize_segments(segmented_markers, show_labels=True)
    filtered_image = visualize_segments(filtered_markers, show_labels=True)

    cv2.imshow("Final Segmentation", final_image)
    cv2.imshow("Filtered Segmentation", filtered_image)
    cv2.waitKey()
    print(np.unique(filtered_markers, return_counts=True))

    np.save("house_segmentation.npy", filtered_markers, allow_pickle=False)
