from Detector import *
import os


def main():

    # videoPath = "Free City Street Footage - Royalty Free Stock Footage - People Walking Stock Footage No Copyright.mp4"
    picturePath = "TEST-2.webp"
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(picturePath, configPath,modelPath, classesPath)
    
    # detector.onVideo()

    detector.onPicture()

if __name__ == '__main__':
    main()
