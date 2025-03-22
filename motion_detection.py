import image_IO
import cv2
import numpy as np
import geometric


def motion_detection_frame_difference(cam, grayscale=True):
    """
        Perform motion detection using frame differencing.

        Args:
            cam (cv2.VideoCapture): OpenCV camera object.
            grayscale (bool): Whether to convert frames to grayscale before processing.

        Displays:
            A binary motion mask highlighting detected motion.
        """
    last = None  # Store the last frame for comparison
    while cam.isOpened():
        ret, frame = cam.read()

        if not ret:
            break  # Stop if no frame is retrieved

        # Apply Gaussian Blur to reduce noise and optionally convert to grayscale
        gray_frame = image_IO.gaussian_blur(frame, grayscale=grayscale)

        if last is not None:
            # Compute absolute frame difference
            diff = np.abs(last.astype(np.int16) - gray_frame.astype(np.int16)).astype(np.uint8)
            # Thresholding to create a motion mask
            motion_mask = np.where(diff > 50, 255, 0).astype(np.uint8)
            # Display the motion mask
            cv2.imshow("Motion Detection", motion_mask)

            # Update last frame
            last = gray_frame
        else:
            last = gray_frame  # Initialize last frame

        # Exit on 'ESC' key press
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break


def process_frame(frame, last_frame, threshold=50, min_cluster_size=1, grayscale=True):
    """
    Process a frame to detect motion using frame differencing.

    Args:
        frame (np.array): Current frame from the camera.
        last_frame (np.array or None): Previous frame for comparison.
        threshold (int): Pixel intensity difference threshold.
        min_cluster_size (int): Minimum size for detected motion clusters.
        grayscale (bool): Whether to convert to grayscale.

    Returns:
        motion_mask (np.array or None): Binary mask of motion.
        merged_box (tuple or None): (x1, y1, x2, y2) of the enclosing box.
        gray_frame (np.array): Processed grayscale frame.
    """
    small_frame = geometric.rescale(frame, 160, 120)  # Resize for faster processing
    gray_frame = image_IO.gaussian_blur(small_frame, grayscale=grayscale)

    if last_frame is None:
        return None, None, gray_frame  # First frame has no previous comparison

    # Compute absolute difference
    diff = np.abs(last_frame.astype(np.int16) - gray_frame.astype(np.int16))
    motion_mask = np.where(diff > threshold, 255, 0).astype(np.uint8)

    # Detect motion clusters if motion is present
    clusters = find_clusters(list(motion_mask), min_size=min_cluster_size) if np.any(motion_mask) else []

    if not clusters:
        return motion_mask, None, gray_frame  # No motion detected

    # 🔹 Merge all motion clusters into a single bounding box
    all_points = np.vstack(clusters)  # Combine all points into one array
    min_y, min_x = all_points.min(axis=0)
    max_y, max_x = all_points.max(axis=0)

    merged_box = (min_x, min_y, max_x, max_y)

    return motion_mask, merged_box, gray_frame


def draw_bounding_box(frame, box, scale_x, scale_y):
    """
    Draw one large bounding box around all detected motion clusters.

    Args:
        frame (np.array): The original full-size frame.
        box (tuple): (x1, y1, x2, y2) coordinates of the bounding box.
        scale_x (float): Scale factor for width.
        scale_y (float): Scale factor for height.
    """
    if not box:
        return  # No motion detected

    # Scale coordinates back to original frame size
    x1, y1 = int(box[0] * scale_x), int(box[1] * scale_y)
    x2, y2 = int(box[2] * scale_x), int(box[3] * scale_y)

    # Draw bounding box in green
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def motion_detection(cam):
    """
    Perform motion detection using frame differencing and bounding box visualization.

    Args:
        cam (cv2.VideoCapture): OpenCV camera object.

    Displays:
        The original frame with a bounding box highlighting motion.
    """
    last_frame = None  # Store previous frame for differencing

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break  # Stop if no frame is retrieved

        # Process frame for motion detection

        motion_mask, merged_box, last_frame = process_frame(frame, last_frame, grayscale=True)

        # Display the original frame
        display_frame = frame.copy()

        # Draw a single large bounding box around all motion
        if merged_box:
            scale_x = frame.shape[1] / 160  # Scale width factor
            scale_y = frame.shape[0] / 120  # Scale height factor
            draw_bounding_box(display_frame, merged_box, scale_x, scale_y)

        # Show the frame with the bounding box
        cv2.imshow("Motion Detection", display_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def find_clusters(matrix, min_size=1):
    """
    Finds clusters of 1s in a binary matrix using DFS with fully optimized NumPy operations.

    Args:
        matrix (list of lists): The input binary matrix.
        min_size (int): The minimum number of 1s required to count a cluster.

    Returns:
        list: A list of clusters, each represented as a list of (row, col) indices.
    """
    matrix = np.array(matrix, dtype=np.uint8)  # Convert to NumPy array
    rows, cols = matrix.shape

    visited = np.zeros(rows * cols, dtype=bool)  # Use a flat array for visited tracking
    clusters = []  # Store valid clusters

    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Up, Down, Left, Right

    # Get all 1's positions at once using np.flatnonzero (faster than np.argwhere)
    ones_indices = np.flatnonzero(matrix.ravel())  # Flatten matrix and get indices

    def dfs(start_idx):
        """ Perform DFS using a NumPy-based stack. """
        stack = np.array([start_idx])  # Initialize stack with the starting point
        clust = []  # Store the indices of the current cluster

        while stack.shape[0] > 0:
            idx = stack[-1]  # Get last element (LIFO)
            stack = stack[:-1].copy()  # Pop last element (copy for efficiency)

            if visited[idx]:  # Skip if already visited
                continue
            visited[idx] = True  # Mark as visited

            row, col = divmod(idx, cols)  # Convert flat index to (row, col)
            clust.append((row, col))  # Add to cluster

            # Compute neighbor indices
            neighbors = directions + [row, col]

            # Filter valid neighbors (inside bounds)
            valid = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < rows) & \
                    (neighbors[:, 1] >= 0) & (neighbors[:, 1] < cols)
            valid_neighbors = neighbors[valid]

            # Convert neighbors to flat indices for fast lookup
            neighbor_indices = np.ravel_multi_index((valid_neighbors[:, 0], valid_neighbors[:, 1]), (rows, cols))

            # Keep only unvisited 1s
            valid_mask = (matrix.ravel()[neighbor_indices] == 1) & (~visited[neighbor_indices])

            if np.any(valid_mask):  # Avoid empty stack operations
                stack = np.concatenate((stack, neighbor_indices[valid_mask]))  # Push valid neighbors

        return clust

    # Process all unvisited 1s
    for ix in ones_indices:
        if not visited[ix]:  # Check if not visited
            cluster = dfs(ix)
            if len(cluster) >= min_size:  # Filter small clusters
                clusters.append(cluster)

    return clusters
