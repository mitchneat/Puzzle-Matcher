import cv2
import numpy as np
import json
import os
from glob import glob
from roboflow import Roboflow
from dotenv import load_dotenv

def infer_roboflow(image_path, api_key):
    print('Running Hole infer on New Model')   
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("puzzle_piece_matcher-branch-2")
    model = project.version(1).model
    result = model.predict(image_path, confidence=80).json()
    return result

def infer_roboflow_oldModel(image_path, api_key):
    print('Running Hole infer on Old Model')
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("puzzle_piece_matcher")
    model = project.version(2).model  
    result = model.predict(image_path, confidence=90).json()
    return result

def extract_polygons_from_json(data, label_name):
    """ Extracts polygon contours from JSON data returned by Roboflow """
    contours = []
    for pred in data.get('predictions', []):
        if pred['class'] == label_name:
            points = np.array([[pt['x'], pt['y']] for pt in pred['points']], dtype=np.int32)
            contours.append(points)
    return contours

def match_shapes_ranked(piece_contours, piece_filenames, hole_contours):
    """ Matches each hole to the best fitting puzzle pieces """
    matches = []
    
    for h_idx, hole in enumerate(hole_contours):
        scores = []
        
        for p_idx, (piece, filename) in enumerate(zip(piece_contours, piece_filenames)):
            score = cv2.matchShapes(piece, hole, cv2.CONTOURS_MATCH_I1, 0.0)
            scores.append((filename, score))
        
        # Sort by lowest score (best match first)
        scores.sort(key=lambda x: x[1])
        
        # Ensure we return top 10 matches if available
        top_matches = scores[:10] if len(scores) >= 10 else scores

        matches.append({
            "hole_id": h_idx,
            "top_matches": [
                {"filename": f"match{rank+1}_{os.path.basename(m[0])}", "score": m[1]} 
                for rank, m in enumerate(top_matches)
            ]
        })

    return matches

def convert_numpy_types(obj):
    """ Converts numpy data types to Python native types for JSON serialization """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj


################################################
################ ---- Main Execution ----
################################################
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("ROBOFLOW_API_KEY")
holes_folder = "./Data/Testing/Holes/"
pieces_folder = "./Data/Testing/Pieces/"
PIECES_JSON_PATH = "./Data/Testing/pieces_data_extracted_oldModel.json" ########################################
# PIECES_JSON_PATH = "./Data/Testing/pieces_data_extracted.json" 


### get data
holes_img = glob(os.path.join(holes_folder, "*.jpg"))[0]
piece_images = glob(os.path.join(pieces_folder, "*.jpg"))
print("Processing hole image:", holes_img)


### run inference
# Run inference on holes
# holes_data = infer_roboflow(holes_img, API_KEY)
holes_data = infer_roboflow_oldModel(holes_img, API_KEY) #################################################

# Run inference on pieces and save results
if not os.path.exists(PIECES_JSON_PATH):
    print("Running inference on puzzle pieces...")
    pieces_data = {img_path: infer_roboflow(img_path, API_KEY) for img_path in piece_images}

    # Save results to JSON file
    with open(PIECES_JSON_PATH, "w") as f:
        json.dump(pieces_data, f, indent=2)
    print("Piece detections saved to", PIECES_JSON_PATH)
else:
    # Load previously saved detections
    with open(PIECES_JSON_PATH, "r") as f:
        pieces_data = json.load(f)
    print("Loaded piece detections from", PIECES_JSON_PATH)


### Extract contours
used_pieces = [75, 4, 36]
hole_contours = extract_polygons_from_json(holes_data, "puzzle-hole")
piece_contours = []
piece_filenames = []

# Make sure each detected contour maps to the correct piece filename
for img_path, data in pieces_data.items():
    piece_number = int(os.path.basename(img_path).replace("img", "").replace(".jpg", ""))
    
    # Skip if the piece is in the used_pieces list
    if piece_number in used_pieces:
        continue

    contours = extract_polygons_from_json(data, "puzzle-piece")
    
    if contours:
        piece_contours.extend(contours)
        piece_filenames.extend([img_path] * len(contours))  


# Match pieces
matches = match_shapes_ranked(piece_contours, piece_filenames, hole_contours)
matches = convert_numpy_types(matches)

# Output the results
print(json.dumps(matches, indent=2))
