"""
Copyright (c) 2020
Hidetoshi Nitta
All rights reserved.
"""

import cv2
import dlib
import imutils
from imutils import face_utils

# 使用するカメラのID 0は標準webカメラ
DEVICE_ID = 0


def main():
    # dlibの学習済みデータの読み込み
    capture = cv2.VideoCapture(DEVICE_ID)
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    while(True):
        # カメラからキャプチャしてframeに１コマ分の画像データを入れる
        ret, frame = capture.read()

        # frameの画像の表示サイズを整える
        frame = imutils.resize(frame, width=1000)
        # gray scaleに変換する
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔を検出
        rects = detector(frame, 0)
        # 顔の検出に失敗するとrects=[]になる
        print(rects)

        for rect in rects:
            # ここでランドマークを検出
            landmark_shape = face_utils.shape_to_np(predictor(frame, rect))
            # 顔全体の68箇所のランドマークをプロット
            for (x, y) in landmark_shape:
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        # 画像を表示する
        cv2.imshow('frame', frame)
        # qを押すとbreakしてwhileから抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # video captureを終了する
    capture.release()
    # windowを閉じる
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
