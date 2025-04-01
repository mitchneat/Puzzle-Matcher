import requests
import cv2
import numpy as np
import json
import tempfile
import os
from roboflow import Roboflow
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import splprep, splev
from dotenv import load_dotenv 

def infer_roboflow(image_path, api_key):
    rf = Roboflow(api_key=api_key)  
    # print(rf.workspace()) # to view workspaces and projects for connection issues
    project = rf.workspace().project("puzzle_piece_matcher-branch-2")
    model = project.version(1).model  

    # Perform inference using Roboflow
    result = model.predict(image_path, confidence=90).json()
    return result

def extract_polygons_from_json(data, label_name):
    contours = []
    for pred in data['predictions']:
        if pred['class'] == label_name:
            points = np.array([[pt['x'], pt['y']] for pt in pred['points']], dtype=np.int32)
            contours.append(points)
    return contours

def smooth_contour(contour):
    # Convert contour to float for better processing
    contour = np.array(contour, dtype=np.float32)

    # Convex Hull to remove noise
    hull = cv2.convexHull(contour)

    # # Fit B-Spline curve for smooth interpolation
    hull = hull.squeeze()  # Remove extra dimension
    # if len(hull) >= 4:  # At least 4 points required for spline
    #     tck, u = splprep([hull[:, 0], hull[:, 1]], s=1)  # Adjust smoothness with 's'
    #     smooth_points = splev(np.linspace(0, 1, len(hull)), tck)
    #     smooth_contour = np.column_stack(smooth_points).astype(np.int32)
    #     return smooth_contour
    # else:
    return hull  # Return as-is if not enough points

# ---- Improved Matching Methods ----
def chamfer_match(piece_contour, hole_contour):
    piece_mask = np.zeros((500, 500), dtype=np.uint8)
    hole_mask = np.zeros((500, 500), dtype=np.uint8)

    cv2.drawContours(piece_mask, [piece_contour], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(hole_mask, [hole_contour], -1, 255, thickness=cv2.FILLED)

    piece_dist = cv2.distanceTransform(255 - piece_mask, cv2.DIST_L2, 3)
    hole_dist = cv2.distanceTransform(255 - hole_mask, cv2.DIST_L2, 3)

    chamfer_distance = np.sum(np.abs(piece_dist - hole_dist))
    return chamfer_distance

# better matching method
def match_shapes(piece_contours, hole_contours):
    n_pieces = len(piece_contours)
    n_holes = len(hole_contours)
    cost_matrix = np.zeros((n_pieces, n_holes))

    # Compute cost matrix
    for i, piece in enumerate(piece_contours):
        for j, hole in enumerate(hole_contours):
            # score = cv2.matchShapes(piece, hole, cv2.CONTOURS_MATCH_I1, 0.0)
            # cost_matrix[i, j] = score
            shape_score = cv2.matchShapes(piece, hole, cv2.CONTOURS_MATCH_I1, 0.0)
            chamfer_score = chamfer_match(piece, hole)
            cost_matrix[i, j] = shape_score + 0.1 * chamfer_score  # Weighted sum

    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_indices, col_indices):
        matches.append({
            "piece_id": i,
            "hole_id": j,
            "score": cost_matrix[i, j]
        })

    return matches

def convert_numpy_types(obj):
    # Recursively convert numpy types to Python types
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar to Python scalar
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

def visualize_matches(holes_img_path, pieces_img_path, matches, piece_contours, hole_contours):
    holes_img = cv2.imread(holes_img_path)
    pieces_img = cv2.imread(pieces_img_path)

    for idx, match in enumerate(matches):
        piece_idx = match['piece_id']
        hole_idx = match['hole_id']

        # Draw contours (optional)
        cv2.polylines(holes_img, [hole_contours[hole_idx]], True, (0, 0, 255), 3)
        cv2.polylines(pieces_img, [piece_contours[piece_idx]], True, (255, 0, 0), 3)

        # Get centers of hole and piece contours
        hole_moments = cv2.moments(hole_contours[hole_idx])
        hole_cx = int(hole_moments['m10'] / hole_moments['m00'])
        hole_cy = int(hole_moments['m01'] / hole_moments['m00'])

        piece_moments = cv2.moments(piece_contours[piece_idx])
        piece_cx = int(piece_moments['m10'] / piece_moments['m00'])
        piece_cy = int(piece_moments['m01'] / piece_moments['m00'])

        # Put bold number at the center of hole and piece
        cv2.putText(holes_img, str(idx + 1), (hole_cx, hole_cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 20, cv2.LINE_AA)
        cv2.putText(pieces_img, str(idx + 1), (piece_cx, piece_cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 20, cv2.LINE_AA)


    # cv2.imshow("Holes Image Matches", holes_img)
    # cv2.imshow("Pieces Image Matches", pieces_img)
    cv2.imwrite("output_holes.jpg", holes_img)
    cv2.imwrite("output_pieces.jpg", pieces_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---- Main Execution ----
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("ROBOFLOW_API_KEY")
holes_img = "./Data/Testing/holes.jpg"
pieces_img = "./Data/Testing/piecesV2.jpg"

# Send preprocessed images to Roboflow
holes_data = infer_roboflow(holes_img, API_KEY)
pieces_data = infer_roboflow(pieces_img, API_KEY)

# Convert numpy types in the results
holes_data = convert_numpy_types(holes_data)
pieces_data = convert_numpy_types(pieces_data)

# Extract contours from Roboflow predictions
hole_contours = extract_polygons_from_json(holes_data, "puzzle-hole")
piece_contours = extract_polygons_from_json(pieces_data, "puzzle-piece")


# hole_contours = [smooth_contour(c) for c in hole_contours] ### bad 
# piece_contours = [smooth_contour(c) for c in piece_contours] ### bad 

# Match the simplified contours
matches = match_shapes(piece_contours, hole_contours) 

matches = convert_numpy_types(matches)
# Output the match results
print("Match Results:")
print(json.dumps(matches, indent=2))

# Visualize the matches on the original images
visualize_matches(holes_img, pieces_img, matches, piece_contours, hole_contours)
