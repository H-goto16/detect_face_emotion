from unicodedata import numeric
from feat import Detector
from feat.utils import get_test_data_path
from feat.plotting import imshow
import os
import json

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='svm',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

test_data_dir = get_test_data_path()

imgPath = "face.png"

while True:
    path = os.path.isfile(imgPath)
    try:
        single_face_img_path = imgPath  
        imshow(single_face_img_path)
        single_face_prediction = detector.detect_image(single_face_img_path)
        list = json.loads(single_face_prediction.to_json())

        anger = list["anger"]["0"]
        disgust = list["disgust"]["0"]
        fear = list["fear"]["0"]
        sadness = list["sadness"]["0"]
        surprise = list["surprise"]["0"]
        neutral = list["neutral"]["0"]
        happiness = list["happiness"]["0"]
        

        emotionDict = {"怒り": anger, "不快": disgust, "恐怖": fear,
                       "驚き": surprise, "普通": neutral, "幸せ": happiness, "悲しみ": sadness }
        

        print([kv for kv in emotionDict.items()
              if kv[1] == max(emotionDict.values())][0])
        
        
        

    except Exception as e:
        pass
