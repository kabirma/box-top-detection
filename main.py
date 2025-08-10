import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.ndimage import label, binary_closing, binary_opening
import time
import pandas as pd
import os

def ransac(points, threshold=0.01, max_iterations=10):
    best_inliers = []
    model = None
    num_points = len(points)
    for _ in range(max_iterations):
        sample_indices = random.sample(range(num_points), 3)
        p1, p2, p3 = points[sample_indices]
        vector1 = p2 - p1
        vector2 = p3 - p1
        normalVector = np.cross(vector1, vector2)
        if np.linalg.norm(normalVector) == 0:
            continue

        unitVector = normalVector / np.linalg.norm(normalVector)
        d = -np.dot(unitVector, p1)
        distances = np.abs(points @ unitVector + d)
        inliers = np.where(distances < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            model = (unitVector, d)

    return model, best_inliers

def mlesac(points, threshold=0.01, max_iterations=100, gamma=None):
    if gamma is None:
        gamma = threshold * 10

    best_model = None
    lowest_cost = float('inf')
    num_points = len(points)

    for _ in range(max_iterations):
        sample_indices = random.sample(range(num_points), 3)
        p1, p2, p3 = points[sample_indices]
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) == 0:
            continue

        normal_unit = normal / np.linalg.norm(normal)
        d = -np.dot(normal_unit, p1)
        distances = np.abs(points @ normal_unit + d)

        cost = np.sum(np.where(distances < threshold, distances, gamma))
        if cost < lowest_cost:
            lowest_cost = cost
            best_model = (normal_unit, d)

    normal_unit, d = best_model
    distances = np.abs(points @ normal_unit + d)
    inliers = np.where(distances < threshold)[0]

    return best_model, inliers

def preemptive_ransac(points, M=100, B=10, threshold=0.01):
    num_points = len(points)
    hypotheses = []

    for _ in range(M):
        sample_indices = random.sample(range(num_points), 3)
        p1, p2, p3 = points[sample_indices]
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) == 0:
            continue
        normal_unit = normal / np.linalg.norm(normal)
        d = -np.dot(normal_unit, p1)
        hypotheses.append((normal_unit, d))

    indices = np.arange(num_points)
    np.random.shuffle(indices)

    i = 0
    while len(hypotheses) > 1 and i < len(indices):
        batch = indices[i:i + B]
        i += B
        scores = []
        for h in hypotheses:
            normal, d = h
            distances = np.abs(points[batch] @ normal + d)
            scores.append(np.sum(distances < threshold))

        sorted_indices = np.argsort(scores)[::-1]
        retain_count = max(1, int(np.floor(len(hypotheses) * 2 ** (-i // B))))
        hypotheses = [hypotheses[i] for i in sorted_indices[:retain_count]]

    best_model = hypotheses[0]
    distances = np.abs(points @ best_model[0] + best_model[1])
    inliers = np.where(distances < threshold)[0]

    return best_model, inliers

def plot(data, title, cmap):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 2)
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_estimators(file_path, amplitude_index, distance_index, cloud_index, thresholds, estimators, export_csv=False, csv_path='results_summary.csv'):
    results = []
    for estimator in estimators:
        for threshold in thresholds:
            print(f"\nEvaluating {estimator.upper()} with threshold={threshold}")
            start_time = time.time()
            box_dims = detect_box_size(
                file_path,
                amplitude_index,
                distance_index,
                cloud_index,
                title=f"{estimator.upper()} with threshold={threshold}",
                estimator=estimator,
                threshold=threshold,
                return_dimensions=True,
                show_plots=False
            )
            duration = time.time() - start_time
            if box_dims:
                height, length, width = box_dims
                results.append({
                    'Estimator': estimator,
                    'Threshold': threshold,
                    'Height_cm': round(height * 100, 3),
                    'Length_cm': round(length * 100, 3),
                    'Width_cm': round(width * 100, 3),
                    'Time_s': round(duration, 2)
                })

    df = pd.DataFrame(results)
    print("\nSummary of Results:")
    print(df)

    df.plot(x='Threshold', y=['Height_cm', 'Length_cm', 'Width_cm'], kind='line', title='Box Dimension Estimates')
    plt.grid(True)
    plt.show()

    df.plot(x='Threshold', y='Time_s', kind='line', title='Computation Time')
    plt.grid(True)
    plt.show()

    if export_csv:
        df.to_csv(csv_path, index=False)
        print(f"Results exported to {os.path.abspath(csv_path)}")

def detect_box_size(file_path, amplitude_index, distance_index, cloud_index, title, estimator='ransac', threshold=0.01, return_dimensions=False, show_plots=True):
    print(title)
    data = scipy.io.loadmat(file_path)
    amplitude = data[amplitude_index]
    distance = data[distance_index]
    cloud = data[cloud_index]

    H, W, _ = cloud.shape
    points = cloud.reshape(-1, 3)
    valid = points[:, 2] != 0
    valid_mask = cloud[:, :, 2] != 0
    valid_points = cloud[valid_mask]

    estimator_fn = {'ransac': ransac, 'mlesac': mlesac, 'preemptive': preemptive_ransac}[estimator]
    floor_model, floor_inliers_idx = estimator_fn(valid_points, threshold=threshold)
    floor_normal, floor_d = floor_model

    floor_mask_flat = np.zeros(valid_points.shape[0], dtype=np.uint8)
    floor_mask_flat[floor_inliers_idx] = 1
    floor_mask_image = np.zeros_like(valid_mask, dtype=np.uint8)
    floor_mask_image[valid_mask] = floor_mask_flat

    floor_mask_image = binary_opening(floor_mask_image, structure=np.ones((3, 3)))
    floor_mask_image = binary_closing(floor_mask_image, structure=np.ones((3, 3)))

    non_floor_mask = (valid_mask.astype(bool) & ~floor_mask_image.astype(bool))
    non_floor_points = cloud[non_floor_mask]
    box_model, box_inliers_idx = estimator_fn(non_floor_points, threshold=threshold)
    box_normal, box_d = box_model

    box_mask_flat = np.zeros(non_floor_points.shape[0], dtype=np.uint8)
    box_mask_flat[box_inliers_idx] = 1
    box_top_mask_image = np.zeros_like(valid_mask, dtype=np.uint8)
    box_top_mask_image[non_floor_mask] = box_mask_flat

    box_top_mask_image = binary_opening(box_top_mask_image, structure=np.ones((3, 3)))
    box_top_mask_image = binary_closing(box_top_mask_image, structure=np.ones((3, 3)))

    labeled, num_features = label(box_top_mask_image)
    if num_features == 0:
        print("No connected component found for box top.")
        return None if return_dimensions else None

    largest_component_label = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
    largest_component_mask = (labeled == largest_component_label)

    box_top_points = cloud[largest_component_mask]
    if np.dot(floor_normal, box_normal) < 0:
        box_normal = -box_normal
        box_d = -box_d

    avg_box_point = np.mean(box_top_points, axis=0)
    height = abs(np.dot(floor_normal, avg_box_point) + floor_d) / np.linalg.norm(floor_normal)

    min_xyz = np.min(box_top_points, axis=0)
    max_xyz = np.max(box_top_points, axis=0)
    length = np.linalg.norm(max_xyz[[0, 1]] - min_xyz[[0, 1]])
    width = np.max([abs(max_xyz[0] - min_xyz[0]), abs(max_xyz[1] - min_xyz[1])])

    print(f"Estimated Box Size:")
    print(f"  Height: {height * 100:.3f} cm")
    print(f"  Length: {length * 100:.3f} cm")
    print(f"  Width : {width * 100:.3f} cm")

    if show_plots:
        plot(amplitude, "Amplitude Image", 'Greens')
        plot(np.clip(distance, 0, 2), "Distance Image", 'Greens')
        plot(np.clip(floor_mask_image, 0, 2), "Floor Mask", 'Greens')
        plot(labeled == largest_component_label, "Box Top (Largest Connected Component)", 'Greens')

    return (height, length, width) if return_dimensions else None


def evaluate_preemptive_vs_others(file_path, amplitude_index, distance_index, cloud_index, thresholds, preemptive_configs, export_csv=False, csv_path='preemptive_results_summary.csv'):
    results = []

    # Evaluate standard RANSAC and MLESAC
    for estimator in ['ransac', 'mlesac']:
        for threshold in thresholds:
            print(f"\nEvaluating {estimator.upper()} with threshold={threshold}")
            start_time = time.time()
            box_dims = detect_box_size(
                file_path,
                amplitude_index,
                distance_index,
                cloud_index,
                title=f"{estimator.upper()} (threshold={threshold})",
                estimator=estimator,
                threshold=threshold,
                return_dimensions=True,
                show_plots=False
            )
            duration = time.time() - start_time
            if box_dims:
                height, length, width = box_dims
                results.append({
                    'Estimator': estimator,
                    'Threshold': threshold,
                    'M': None,
                    'B': None,
                    'Height_cm': round(height * 100, 3),
                    'Length_cm': round(length * 100, 3),
                    'Width_cm': round(width * 100, 3),
                    'Time_s': round(duration, 2)
                })

    # Evaluate preemptive RANSAC with different M and B
    for (M, B) in preemptive_configs:
        for threshold in thresholds:
            print(f"\nEvaluating PREEMPTIVE RANSAC with M={M}, B={B}, threshold={threshold}")
            start_time = time.time()
            box_dims = detect_box_size(
                file_path,
                amplitude_index,
                distance_index,
                cloud_index,
                title=f"Preemptive RANSAC (M={M}, B={B}, threshold={threshold})",
                estimator='preemptive',
                threshold=threshold,
                return_dimensions=True,
                show_plots=False
            )
            duration = time.time() - start_time
            if box_dims:
                height, length, width = box_dims
                results.append({
                    'Estimator': 'preemptive',
                    'Threshold': threshold,
                    'M': M,
                    'B': B,
                    'Height_cm': round(height * 100, 3),
                    'Length_cm': round(length * 100, 3),
                    'Width_cm': round(width * 100, 3),
                    'Time_s': round(duration, 2)
                })

    df = pd.DataFrame(results)
    print("\nSummary of Results:")
    print(df)

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="whitegrid")

    # Plot dimensions
    for dim in ['Height_cm', 'Length_cm', 'Width_cm']:
        plt.figure()
        sns.lineplot(data=df, x='Threshold', y=dim, hue='Estimator', style='Estimator', markers=True, dashes=False)
        plt.title(f'Comparison of {dim}')
        plt.grid(True)
        plt.show()

    # Plot time
    plt.figure()
    sns.lineplot(data=df, x='Threshold', y='Time_s', hue='Estimator', style='Estimator', markers=True, dashes=False)
    plt.title('Computation Time')
    plt.grid(True)
    plt.show()

    if export_csv:
        df.to_csv(csv_path, index=False)
        print(f"Results exported to {os.path.abspath(csv_path)}")




# I am only using this on single data set to check results and performance between different variations
evaluate_preemptive_vs_others(
    file_path='./data/example1kinect.mat',
    amplitude_index='amplitudes1',
    distance_index='distances1',
    cloud_index='cloud1',
    thresholds=[0.005, 0.01, 0.015],
    preemptive_configs=[(100, 10), (150, 20), (200, 30)],
    export_csv=True
)
