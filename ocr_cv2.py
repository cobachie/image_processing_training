import sys

import cv2
import numpy as np
from PIL import Image
import pyocr

def mean_hsv_in_roi(image, position):
  """
  ROI の HSV 平均値を取得する
  """

  # TODO: ROI の範囲は要調整
  top, bottom = (position[1][1] + 120), (position[1][1] + 150)
  left, right = position[0][0], position[0][0] + 100
  roi = image[top: bottom, left: right]
  # cv2.rectangle(hsv, (left, top), (right, bottom), (255, 0, 0))

  # ROI の HSV 平均値を算出
  h = roi.T[0].flatten().mean()
  s = roi.T[1].flatten().mean()
  v = roi.T[2].flatten().mean()

  return (h, s, v)


def judge_color(h, s, v):
  """
  HSV値から Red or Yello or Other(Green) を判定する
  """

  # TODO: 境界値は要調整
  color = 'Green'
  if (h <= 30) or (h >= 150):
    color = 'Red'
  elif (30 < h < 60) and (s > 100):
    color = 'Yellow'

  return color


def main():

  tools = pyocr.get_available_tools()
  if len(tools) == 0:
    print('No OCR tool found')
    sys.exit(1)

  tool = tools[0]
  print("Will use tool '%s'" % (tool.get_name()))
  # Ex: Will use tool 'Tesseract (sh)'

  # 利用できる言語を確認
  langs = tool.get_available_languages()
  print("Available languages: %s" % ",".join(langs))
  lang = langs[0]
  print("Will use lang '%s'" % (lang))

  input_image_path = 'images/hotel.png'

  # 部屋番号の位置情報を取得する
  image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
  # TODO: 画像の前処理

  # 部屋番号と座標を取得
  word_boxes = tool.image_to_string(
      Image.fromarray(image),
      lang=lang,
      builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6)
  )

  # 画像を HSV 色空間に変換
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  room_status = {}
  for word in word_boxes:
    w = word.content[-4:]
    if w.isdigit() and len(w) == 4:
      h, s, v = mean_hsv_in_roi(hsv, word.position)
      color = judge_color(h, s, v)
      # print(w, color, h, s, v)
      room_status[w] = color

  # print(room_status)
  return room_status

if __name__ == "__main__":
  main()
