from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict
from datetime import datetime

<<<<<<< HEAD
=======
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
# import cv2
# import numpy as np
# import mediapipe as mp
import math
import io

app = FastAPI()

>>>>>>> 4a6492e9cf13d04f871c45a888812a63a8fabeb5

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh

# def calculate_symmetry(image) -> Dict:
#     results = {}
#     with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         output = face_mesh.process(rgb)

#         if not output.multi_face_landmarks:
#             return {"error": "No face detected"}

#         landmarks = output.multi_face_landmarks[0].landmark
#         h, w, _ = image.shape
#         points = [(int(l.x * w), int(l.y * h)) for l in landmarks]

#         # Example: Compare left eye vs right eye width
#         left_eye = np.linalg.norm(np.array(points[133]) - np.array(points[33]))
#         right_eye = np.linalg.norm(np.array(points[362]) - np.array(points[263]))
#         eye_symmetry = (min(left_eye, right_eye) / max(left_eye, right_eye)) * 100

#         # Example: Compare lips
#         left_lip = points[61]
#         right_lip = points[291]
#         lip_width = np.linalg.norm(np.array(left_lip) - np.array(right_lip))
#         mid_lip = points[13]
#         symmetry_center = abs((left_lip[0] + right_lip[0]) / 2 - mid_lip[0])
#         lip_symmetry = 100 - (symmetry_center / lip_width * 100)

#         results = {
#             "eye_symmetry": round(eye_symmetry, 2),
#             "lip_symmetry": round(lip_symmetry, 2),
#             "overall_symmetry": round((eye_symmetry + lip_symmetry) / 2, 2)
#         }
#     return results

# def calculate_symmetry(image) -> Dict:
#     results = {}
#     with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         output = face_mesh.process(rgb)

#         if not output.multi_face_landmarks:
#             return {"error": "No face detected"}

#         landmarks = output.multi_face_landmarks[0].landmark
#         h, w, _ = image.shape
#         points = [(int(l.x * w), int(l.y * h)) for l in landmarks]

#         def ratio(val1, val2):
#             return (min(val1, val2) / max(val1, val2)) * 100

#         # 1. Eye Symmetry
#         left_eye = np.linalg.norm(np.array(points[133]) - np.array(points[33]))
#         right_eye = np.linalg.norm(np.array(points[362]) - np.array(points[263]))
#         eye_symmetry = ratio(left_eye, right_eye)

#         # 2. Eyebrow Symmetry
#         left_brow = np.linalg.norm(np.array(points[70]) - np.array(points[105]))
#         right_brow = np.linalg.norm(np.array(points[336]) - np.array(points[334]))
#         eyebrow_symmetry = ratio(left_brow, right_brow)

#         # 3. Lip & Mouth Symmetry
#         left_lip = points[61]
#         right_lip = points[291]
#         lip_width = np.linalg.norm(np.array(left_lip) - np.array(right_lip))
#         mid_lip = points[13]
#         symmetry_center = abs((left_lip[0] + right_lip[0]) / 2 - mid_lip[0])
#         lip_symmetry = 100 - (symmetry_center / lip_width * 100)

#         # 4. Nose Symmetry
#         nose_left = points[98]
#         nose_right = points[327]
#         nose_center = points[1]
#         nose_symmetry = 100 - (abs((nose_left[0] + nose_right[0]) / 2 - nose_center[0]) / (nose_right[0] - nose_left[0]) * 100)

#         # 5. Jawline & Chin Symmetry
#         chin_left = points[172]
#         chin_right = points[397]
#         chin_center = points[152]
#         jaw_symmetry = 100 - (abs((chin_left[0] + chin_right[0]) / 2 - chin_center[0]) / (chin_right[0] - chin_left[0]) * 100)

#         # 6. Cheek & Contour Symmetry
#         cheek_left = points[234]
#         cheek_right = points[454]
#         cheek_center = points[1]
#         cheek_symmetry = 100 - (abs((cheek_left[0] + cheek_right[0]) / 2 - cheek_center[0]) / (cheek_right[0] - cheek_left[0]) * 100)

#         results = {
#             "Cheek & Contour Symmetry Evaluation": round(cheek_symmetry, 2),
#             "Jawline & Chin Symmetry Analysis": round(jaw_symmetry, 2),
#             "Lip & Mouth Symmetry Test": round(lip_symmetry, 2),
#             "Nose Symmetry Analysis": round(nose_symmetry, 2),
#             "Eyebrow Symmetry Test": round(eyebrow_symmetry, 2),
#             "Eye Symmetry Test": round(eye_symmetry, 2),
#         }

#     return results




def calculate_symmetry(image) -> Dict:
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb)

        if not output.multi_face_landmarks:
            return {"error": "No face detected"}

        landmarks = output.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        points = [(int(l.x * w), int(l.y * h)) for l in landmarks]

        def ratio(val1, val2):
            return (min(val1, val2) / max(val1, val2)) * 100

        # --- Feature calculations ---
        # Eye Symmetry
        left_eye = np.linalg.norm(np.array(points[133]) - np.array(points[33]))
        right_eye = np.linalg.norm(np.array(points[362]) - np.array(points[263]))
        eye_symmetry = ratio(left_eye, right_eye)

        # Eyebrow Symmetry
        left_brow = np.linalg.norm(np.array(points[70]) - np.array(points[105]))
        right_brow = np.linalg.norm(np.array(points[336]) - np.array(points[334]))
        eyebrow_symmetry = ratio(left_brow, right_brow)

        # Lip Symmetry
        left_lip = points[61]
        right_lip = points[291]
        lip_width = np.linalg.norm(np.array(left_lip) - np.array(right_lip))
        mid_lip = points[13]
        symmetry_center = abs((left_lip[0] + right_lip[0]) / 2 - mid_lip[0])
        lip_symmetry = 100 - (symmetry_center / lip_width * 100)

        # Nose Symmetry
        nose_left = points[98]
        nose_right = points[327]
        nose_center = points[1]
        nose_symmetry = 100 - (abs((nose_left[0] + nose_right[0]) / 2 - nose_center[0]) / (nose_right[0] - nose_left[0]) * 100)

        # Jawline Symmetry
        chin_left = points[172]
        chin_right = points[397]
        chin_center = points[152]
        jaw_symmetry = 100 - (abs((chin_left[0] + chin_right[0]) / 2 - chin_center[0]) / (chin_right[0] - chin_left[0]) * 100)

        # Cheek Symmetry
        cheek_left = points[234]
        cheek_right = points[454]
        cheek_center = points[1]
        cheek_symmetry = 100 - (abs((cheek_left[0] + cheek_right[0]) / 2 - cheek_center[0]) / (cheek_right[0] - cheek_left[0]) * 100)

        # --- Feature scores dictionary ---
        feature_scores = {
            "eyes": round(eye_symmetry, 2),
            "eyebrows": round(eyebrow_symmetry, 2),
            "nose": round(nose_symmetry, 2),
            "lips": round(lip_symmetry, 2),
            "jawline": round(jaw_symmetry, 2),
            "cheeks": round(cheek_symmetry, 2),
        }

        # Overall average
        overall_score = sum(feature_scores.values()) / len(feature_scores)

        # --- Dominant Shape Logic ---
        if overall_score >= 95:
            dominant = "perfect"
        elif overall_score >= 85:
            dominant = "nearPerfect"
        elif overall_score >= 70:
            dominant = "slight"
        elif overall_score >= 55:
            dominant = "moderate"
        elif overall_score >= 40:
            dominant = "significant"
        else:
            dominant = "strong"

        # scores = {
        #     "perfect": 95 if dominant == "perfect" else 0,
        #     "nearPerfect": 90 if dominant == "nearPerfect" else 0,
        #     "slight": 70 if dominant == "slight" else 0,
        #     "moderate": 55 if dominant == "moderate" else 0,
        #     "significant": 40 if dominant == "significant" else 0,
        #     "strong": 20 if dominant == "strong" else 0,
        # }

        scores = {
    "perfect": overall_score if dominant == "perfect" else 0,
    "nearPerfect": overall_score if dominant == "nearPerfect" else 0,
    "slight": overall_score if dominant == "slight" else 0,
    "moderate": overall_score if dominant == "moderate" else 0,
    "significant": overall_score if dominant == "significant" else 0,
    "strong": overall_score if dominant == "strong" else 0,
}


        # --- Analysis text ---
        analysis = (
            f"The face exhibits {dominant.replace('nearPerfect','near-perfect')} symmetry. "
            f"The eyes are aligned (score {feature_scores['eyes']}). "
            f"The eyebrows are balanced (score {feature_scores['eyebrows']}). "
            f"The nose is midline-aligned (score {feature_scores['nose']}). "
            f"The lips are symmetrical (score {feature_scores['lips']}). "
            f"The jawline and chin are even (score {feature_scores['jawline']}). "
            f"The cheek contours are balanced (score {feature_scores['cheeks']})."
        )

        return {
            "dominantShape": dominant,
            "scores": scores,
            "featureScores": feature_scores,
            "analysis": analysis,
            "analysisTime": datetime.utcnow().isoformat() + "Z"
        }


@app.post("/symmetry")
async def get_symmetry(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = calculate_symmetry(img)
    return result







# {
#     "dominantShape": "nearPerfect",
#     "scores": {
#         "perfect": 80,
#         "nearPerfect": 90,
#         "slight": 0,
#         "moderate": 0,
#         "significant": 0,
#         "strong": 0
#     },
#     "featureScores": {
#         "eyes": 90,
#         "eyebrows": 85,
#         "nose": 95,
#         "lips": 80,
#         "jawline": 85,
#         "cheeks": 90
#     },
#     "analysis": "The face exhibits near-perfect symmetry, with the left and right sides being almost identical. The eyes are well-aligned, with consistent position, size, and shape. The eyebrows are balanced in height and curvature. The nose is midline-aligned with the nasal bridge and nostrils. The lips are symmetrical in terms of corner height and shape. The jawline and chin are also symmetrical. The cheek contours and cheekbones are well-balanced, contributing to the overall harmony of the face.",
#     "analysisTime": "2025-09-07T07:21:12.688Z"
# }



# {
#     "dominantShape": "nearPerfect",
#     "scores": {
#         "perfect": 0,
#         "nearPerfect": 90,
#         "slight": 0,
#         "moderate": 0,
#         "significant": 0,
#         "strong": 0
#     },
#     "featureScores": {
#         "eyes": 86.48,
#         "eyebrows": 73.02,
#         "nose": 96.15,
#         "lips": 97.22,
#         "jawline": 94.71,
#         "cheeks": 93.69
#     },
#     "analysis": "The face exhibits near-perfect symmetry. The eyes are aligned (score 86.48). The eyebrows are balanced (score 73.02). The nose is midline-aligned (score 96.15). The lips are symmetrical (score 97.22). The jawline and chin are even (score 94.71). The cheek contours are balanced (score 93.69).",
#     "analysisTime": "2025-09-07T07:23:59.544316Z"
<<<<<<< HEAD
# }   
=======
# }   



mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Euclidean distance
def distance(p1, p2):
    return math.dist(p1, p2)

# Function to calculate golden ratio % match
def calculate_golden_ratio(landmarks, img_w, img_h):
    # Convert relative landmarks to pixel coords
    pts = [(int(l.x * img_w), int(l.y * img_h)) for l in landmarks]

    # Example golden ratio checks (can extend with more rules)
    # 1. Face length / Face width
    top_head = pts[10]   # forehead
    chin = pts[152]      # chin
    left_cheek = pts[234]
    right_cheek = pts[454]

    face_length = distance(top_head, chin)
    face_width = distance(left_cheek, right_cheek)
    ratio1 = face_length / face_width if face_width != 0 else 0

    # 2. Eye to lip vs chin distance
    left_eye = pts[33]
    right_eye = pts[263]
    mid_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    upper_lip = pts[13]

    eye_lip = distance(mid_eye, upper_lip)
    lip_chin = distance(upper_lip, chin)
    ratio2 = lip_chin / eye_lip if eye_lip != 0 else 0

    # Compare with golden ratio
    golden = 1.618
    score1 = (1 - abs(ratio1 - golden) / golden) * 100
    score2 = (1 - abs(ratio2 - golden) / golden) * 100

    return {
        "face_length_width_ratio": round(ratio1, 3),
        "eye_lip_chin_ratio": round(ratio2, 3),
        "golden_ratio_match_percentage": round((score1 + score2) / 2, 2)
    }

@app.post("/analyze-face/")
async def analyze_face(file: UploadFile = File(...)):
    try:
        # Read image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_h, img_w, _ = img.shape

        # Detect face landmarks
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                return JSONResponse({
                    "status": False,
                    "message": "Not proper face image"
                })

            landmarks = results.multi_face_landmarks[0].landmark
            analysis = calculate_golden_ratio(landmarks, img_w, img_h)

            return JSONResponse({
                "status": True,
                "message": "Face analyzed successfully",
                "data": analysis
            })

    except Exception as e:
        return JSONResponse({
            "status": False,
            "message": f"Error processing image: {str(e)}"
        })
>>>>>>> 4a6492e9cf13d04f871c45a888812a63a8fabeb5
