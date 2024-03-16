#!/usr/bin/env python

import sys
import os
import argparse
import cv2
import numpy as np

# 行毎に標準出力のバッファをflushする。
sys.stdout.reconfigure(line_buffering=True)

# 自分の環境のATOM CamのIPに修正してください。
ATOM_CAM_IP = os.environ.get("ATOM_CAM_IP", "192.168.100.13")


def composite(list_images):
    """画像リストの合成(単純スタッキング)

    Args:
      list_images: 画像データのリスト

    Returns:
      合成された画像
    """
    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)

    return output


def median(list_images, opencl=False):
    img_list = []
    if opencl:
        for img in list_images:
            img_list.append(cv2.UMat.get(img))
    else:
        for img in list_images:
            img_list.append(img)

    return np.median (img_list, axis=0).astype(np.uint8)


def average(list_images, opencl=False):
    img_list = []
    if opencl:
        for img in list_images:
            img_list.append(cv2.UMat.get(img))
    else:
        for img in list_images:
            img_list.append(img)

    return np.average(img_list, axis=0).astype(np.uint8)


def brightest(img_list):
    """比較明合成処理
    Args:
      img_list: 画像データのリスト

    Returns:
      比較明合成された画像
    """
    output = img_list[0]

    for img in img_list[1:]:
        output = cv2.max(img, output)

    return output


def diff(img_list, mask):
    """画像リストから差分画像のリストを作成する。

    Args:
      img_list: 画像データのリスト
      mask: マスク画像(2値画像)

    Returns:
      差分画像のリスト
    """
    diff_list = []
    for img1, img2 in zip(img_list[:-2], img_list[1:]):
        if mask is not None:
            img1 = cv2.bitwise_or(img1, mask)
            img2 = cv2.bitwise_or(img2, mask)
        diff_list.append(cv2.subtract(img1, img2))

    return diff_list


def detect(img, min_length):
    """画像上の線状のパターンを流星として検出する。
    Args:
      img: 検出対象となる画像
      min_length: HoughLinesPで検出する最短長(ピクセル)
    Returns:
      検出結果
    """
    blur_size = (5, 5)
    blur = cv2.GaussianBlur(img, blur_size, 0)
    canny = cv2.Canny(blur, 100, 200, 3)

    # The Hough-transform algo:
    return canny, cv2.HoughLinesP(canny, 1, np.pi/180, 25, minLineLength=min_length, maxLineGap=5)


if __name__ == '__main__':

    #from atomcam_streaming import streaming_thread
    from atomcam_videofile import detect_meteor

    parser = argparse.ArgumentParser(add_help=False)

    # ストリーミングモードのオプション
    #parser.add_argument('-u', '--url', default=None,
    #                    help='RTSPのURL、または動画(MP4)ファイル')
    #parser.add_argument('-n', '--no_window', action='store_true', help='画面非表示')
    # threadモード(default)
    #parser.add_argument('--thread', default=True,
    #                    action='store_true', help='スレッド版(default)')
    # 以下のオプションはatomcam_toolsを必要とする。
    #parser.add_argument(
    #    '--atomcam_tools', action='store_true', help='atomcam_toolsを使う場合に指定する。')
    #parser.add_argument(
    #    '-c', '--clock', action='store_true', help='カメラの時刻チェック(atomcam_tools必要)')
    #parser.add_argument('-t', '--to', default="0600",
    #                    help='終了時刻(JST) "hhmm" 形式(ex. 0600)')

    # 以下はATOM Cam形式のディレクトリからデータを読む場合のオプション
    parser.add_argument('-d', '--date', default=None,
                        help="Date in 'yyyymmdd' format (JST)")
    parser.add_argument('-h', '--hour', default=None,
                        help="Hour in 'hh' format (JST)")
    parser.add_argument('-m', '--minute', default=None,
                        help="minute in mm (optional)")
    parser.add_argument('-i', '--input', default=None, help='検出対象のTOPディレクトリ名')
    parser.add_argument(
        '--rectangle', action='store_true', default=False, help="mark HoghLinesP detect lines")
    parser.add_argument(
        '--cannyedge', action='store_true', default=False, help="mark canny edge")

    # 共通オプション
    parser.add_argument('-e', '--exposure', type=int,
                        default=1, help='露出時間(second)')
    parser.add_argument('-o', '--output', default=None, help='検出画像の出力先ディレクトリ名')

    parser.add_argument('--mask', default=None, help="mask image")
    parser.add_argument('--min_length', type=int, default=30,
                        help="minLineLength of HoghLinesP")

    parser.add_argument('--opencl',
                        action='store_true', help="Use OpenCL (default: False)")

    # ffmpeg関係の警告がウザいので抑制する。
    parser.add_argument('-s', '--suppress-warning',
                        action='store_true', help='suppress warning messages')

    parser.add_argument('--help', action='help',
                        help='show this help message and exit')

    args = parser.parse_args()

    if args.suppress_warning:
        # stderrを dev/null に出力する。
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, 2)

    detect_meteor(args)

    #if args.date:
        # 日付がある場合はファイル(ATOMCam形式のファイル)から流星検出
    #    detect_meteor(args)
    #else:
        # ストリーミング/動画(MP4)の再生、流星検出
    #    streaming_thread(args)
