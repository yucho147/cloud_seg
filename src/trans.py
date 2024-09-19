import numpy as np
import cv2


# シグモイド関数の定義
def sigmoid(a):
    return 1. / (1. + np.exp(-a))


# モルフォロジー変換(Opting/ノイズ除去)
def morph_open(a, kernel=(3, 3)):
    kernel = np.ones(kernel, np.uint8)
    return cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)


# モルフォロジー変換(Closing/穴除去)
def morph_close(a, kernel=(3, 3)):
    kernel = np.ones(kernel, np.uint8)
    return cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)


# モルフォロジー変換(Closing + Opening/穴除去)
def morph_close_open(a, kernel=(3, 3)):
    kernel = np.ones(kernel, np.uint8)
    return morph_open(morph_close(a, kernel), kernel)
