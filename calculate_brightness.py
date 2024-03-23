import argparse
import os
import re
from statistics import median
import cv2


def calculate_video_file_brightness(video_file_path, mask_path=None, silent=False):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        return
    ret, frame = cap.read()
    if mask_path is not None:
        mask = cv2.imread(mask_path)
        image = cv2.bitwise_or(frame, mask)
    else:
        image = frame
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = cv2.mean(gray_image)[0]
    cap.release()
    if not silent:
        print(f"{video_file_path} brightness: {average_brightness}")
    return average_brightness

def calculate_atomcam_hour_folder_brightness(input_path, mask=None, silent=False):

    filenames = [n for n in os.listdir(input_path) if n.endswith('.mp4')]
    filenames.sort()
    calc_list = []
    for filename in filenames:
        filepath = os.path.join(input_path, filename,)
        try:
            average_brightness = calculate_video_file_brightness(filepath, mask, silent=silent)
            calc_list.append(average_brightness)
        except Exception as e:
            print(e)
            continue

    value = median(calc_list) if len(calc_list)>0 else None
    print(f"{input_path} average brightness: {value}")
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-d', '--date', default=None,
                        help="Date in 'yyyymmdd' format (JST)")
    parser.add_argument('-h', '--hour', default=None,
                        help="Hour in 'hh' format (JST)")
    parser.add_argument('-m', '--minute', default=None,
                        help="minute in mm (optional)")
    parser.add_argument('-i', '--input', default=None, help='検出対象のTOPディレクトリ名')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    base_path = args.input if args.input is not None else "."

    if not args.date:
        print("dateを指定してください")
    else:
        path = os.path.join(base_path, args.date)
        if args.hour:
            if args.minute:
                calculate_video_file_brightness(os.path.join(path, args.hour, f"{args.minute}.mp4"))
            else:
                calculate_atomcam_hour_folder_brightness(os.path.join(path, args.hour), silent=not args.verbose)
        else:
            dirnames = [d for d in os.listdir(path) if re.match("^\d\d$", d)]
            dirnames.sort()
            for h in dirnames:
                calculate_atomcam_hour_folder_brightness(os.path.join(path, h), silent=not args.verbose)


    