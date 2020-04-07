import sys

import cv2
from PIL import Image
import pyocr

def room_number_position(tool, lang, input_image_path):
  """
  画像内の部屋番号の位置情報を取得する
  """

  positions = {}

  with Image.open(input_image_path) as image:
    # HSV色空間に変換
    hsv_image = image.convert("HSV")

    word_boxes = tool.image_to_string(
        hsv_image,
        lang=lang,
        builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6)
    )

    # 部屋番号と座標を取得
    for word in word_boxes:
      w = word.content[-4:]
      if w.isdigit() and len(w) == 4:
        positions[w] = word.position

  return positions


def mean_rgb(image, position):
  """
  部屋番号の下の色領域を取得する
  """

  top, bottom = (position[1][1] + 120), (position[1][1] + 150)
  left, right = position[1][0] - 100, position[1][0]
  image_box = image[top: bottom, left: right]

  # ROI の RGB 平均値を取得
  b = image_box.T[0].flatten().mean()
  g = image_box.T[1].flatten().mean()
  r = image_box.T[2].flatten().mean()

  # Debug
  # image = cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0))

  return (r, g, b)


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
  positions = room_number_position(tool, lang, input_image_path)

  # 色検出のため cv2 で再度画像読み込み
  image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

  room_status = {}
  for num, position in positions.items():
    # 部屋ごとの RGB を取得する
    room_status[num] = mean_rgb(image, position)

  print(room_status)

  # Debug
  # cv2.imshow('test', image)
  # cv2.waitKey(0)

if __name__ == "__main__":
  main()
