from datetime import datetime
import os
from pathlib import Path
import sys
import cv2
from imutils.video import FileVideoStream
import numpy as np

from atomcam import brightest, detect, diff



def calculate_moving_average(data, window_size):
    moving_averages = []
    
    # データ長がウィンドウサイズより小さい場合は処理不可なのでエラーを返す
    if len(data) < window_size:
        raise ValueError("Data length must be greater than or equal to window size.")
        
    # 移動平均を計算
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        average = np.mean(window)
        moving_averages.append(average)
        
    return moving_averages

class DetectMeteor():
    """
    ATOMCam 動画ファイル(MP4)からの流星の検出
    親クラスから継承したものにしたい。
    """

    def __init__(self, file_path, mask=None, minLineLength=30, opencl=False, rectangle=False, cannyedge=False):
        # video device url or movie file path
        self.capture = FileVideoStream(file_path).start()
        self.HEIGHT = int(self.capture.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH = int(self.capture.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.FPS = self.capture.stream.get(cv2.CAP_PROP_FPS)
        self.source = None
        self.opencl = opencl
        if self.FPS < 1.0:
            # 正しく入っていない場合があるので、その場合は15固定にする(ATOM Cam限定)。
            self.FPS = 15
        self.rectangle = rectangle
        self.cannyedge = cannyedge

        # file_pathから日付、時刻を取得する。
        # date_element = file_path.split('/')
        date_element = file_path.split(os.sep)
        self.date_dir = date_element[-3]
        self.date = datetime.strptime(self.date_dir, "%Y%m%d")

        self.hour = date_element[-2]
        self.minute = date_element[-1].split('.')[0]
        self.obs_time = "{}/{:02}/{:02} {}:{}".format(
            self.date.year, self.date.month, self.date.day, self.hour, self.minute)

        if mask:
            # マスク画像指定の場合
            self.mask = cv2.imread(mask)
        else:
            # 時刻表示部分のマスクを作成
            if opencl:
                zero = cv2.UMat((1080, 1920), cv2.CV_8UC3)
            else:
                zero = np.zeros((1080, 1920, 3), np.uint8)
            if self.source == "Subaru":
                # mask SUBRU/Mauna-Kea timestamp
                self.mask = cv2.rectangle(
                    zero, (1660, 980), (1920, 1080), (255, 255, 255), -1)
            else:
                # mask ATOM Cam timestamp
                self.mask = cv2.rectangle(
                    zero, (1390, 1010), (1920, 1080), (255, 255, 255), -1)

        self.min_length = minLineLength
        self.brightness_values = []

    def save_movie(self, img_list, pathname):
        """
        画像リストから動画を作成する。

        Args:
          imt_list: 画像のリスト
          pathname: 出力ファイル名
        """
        size = (self.WIDTH, self.HEIGHT)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        video = cv2.VideoWriter(pathname, fourcc, self.FPS, size)
        for img in img_list:
            video.write(img)

        video.release()

    def calculate_brightness(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_brightness = cv2.mean(gray_image)[0]
        return average_brightness

    def meteor(self, exposure=1, output=None):
        """流星の検出
        """
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = Path('.')

        num_frames = int(self.FPS * exposure)
        composite_img = None

        count = 0
        while self.capture.more():
            img_list = []
            for n in range(num_frames):
                try:
                    if self.capture.more():
                        frame = self.capture.read()
                        if self.opencl:
                            frame = cv2.UMat(frame)
                    else:
                        continue
                except Exception as e:
                    print(e, file=sys.stderr)
                    continue

                img_list.append(frame)

            self.brightness_values.append(self.calculate_brightness(img_list[0]))

            # 画像のコンポジット
            number = len(img_list)
            count += 1

            # print(number, num_frames)
            if number > 2:
                try:
                    diff_img = brightest(diff(img_list, self.mask))
                    canny, detect_list = detect(diff_img, self.min_length) 
                    if detect_list is None or detect_list.size == 0:
                        continue
                    obs_time = "{}:{}".format(
                        self.obs_time, str(count*exposure).zfill(2))
                    print('{}  A possible meteor was detected.'.format(obs_time))
                    filename = self.date_dir + self.hour + \
                        self.minute + str(count*exposure).zfill(2)
                    path_name = str(Path(output_dir, filename + ".jpg"))
                    path_name2 = str(Path(output_dir, filename + "_canny.jpg"))
                    path_name3 = str(Path(output_dir, filename + "_rect.jpg"))
                    # cv2.imwrite(filename + "_diff.jpg", diff_img)
                    composite_img = brightest(img_list)
                        
                    cv2.imwrite(path_name, composite_img)

                    if self.rectangle:
                        for d in detect_list:
                            cv2.rectangle(composite_img, (d[0][0],d[0][1]), (d[0][2],d[0][3]), (0, 0, 255), 3) 
                        cv2.imwrite(path_name3, composite_img)
                    if self.cannyedge:
                        cv2.imwrite(path_name2, canny)

                    if not self.cannyedge:
                        # 検出した動画を保存する。
                        movie_file = str(
                            Path(output_dir, "movie-" + filename + ".mp4"))
                        self.save_movie(img_list, movie_file)

                except Exception as e:
                    # print(traceback.format_exc(), file=sys.stderr)
                    print(e, file=sys.stderr)
    
    def next_detect_file_or_terminate(self):
        BRIGHTNESS_THRESHOLD = 103
        MOVING_AVERAGE_WINDOW = 10
        SKIP_MINUTES = 15
        moving_avg = calculate_moving_average(self.brightness_values, MOVING_AVERAGE_WINDOW)
        if sum(1 for x in moving_avg if x >= BRIGHTNESS_THRESHOLD) > len(moving_avg)/2:
            print(f"{self.obs_time} Brightness has exceeded the threshold.")
            next = None
            if int(self.hour) > 12:
                if int(self.minute) < 60-SKIP_MINUTES:
                    next = int(self.minute) + SKIP_MINUTES 
                    print("skip next file {:02}.mp4".format(next))
        else:
            next = int(self.minute) + 1
        return "{:02}".format(next) if next else next



def detect_meteor(args):
    """
    ATOM Cam形式の動画ファイルからの流星の検出
    """
    if args.input:
        # 入力ファイルのディレクトリの指定がある場合
        input_dir = Path(args.input)
    else:
        input_dir = Path('.')

    if not args.date:
        print("atomcam保存形式で保存された動画ファイルの日付指定は必須です")
        return

    data_dir = Path(input_dir, args.date)
    if args.hour:
        # 時刻(hour)の指定がある場合
        data_dir = Path(data_dir, args.hour)
        if args.minute:
            # 1分間のファイル単体の処理
            file_path = Path(data_dir, "{}.mp4".format(args.minute))

    print("# {}".format(data_dir))

    if args.minute:
        # 1分間の単体のmp4ファイルの処理
        print("#", file_path)
        detecter = DetectMeteor(
            str(file_path), mask=args.mask, minLineLength=args.min_length, rectangle=args.rectangle, cannyedge=args.cannyedge)
        detecter.meteor(args.exposure, args.output)
    else:
        # 1時間内の一括処理
        file_list = sorted(Path(data_dir).glob("[0-9][0-9].mp4"))
        next = os.path.basename(file_list[0])[:2]
        for file_path in file_list:
            if f"{next}.mp4" != os.path.basename(file_path):
                continue
            print('#', Path(file_path))
            detecter = DetectMeteor(str(file_path), args.mask, rectangle=args.rectangle, cannyedge=args.cannyedge)
            detecter.meteor(args.exposure, args.output)
            next = detecter.next_detect_file_or_terminate()
            if next is None:
                break
