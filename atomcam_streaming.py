
import datetime
import sys
import cv2
from pathlib import Path
import numpy as np
from atomcam import ATOM_CAM_IP, average, brightest, detect, diff
from atomutil import check_clock
try:
    import apafy as pafy
except Exception:
    # pafyを使う場合はpacheが必要。
    import pafy

# マルチスレッド関係
import threading
import queue

import traceback


# YouTube ライブ配信ソース (変更になった場合は要修正)
YouTube = {
    "SDRS6JQulmI": "Kiso",
    "_8rp1p_tWlc": "Subaru",
    "ylSiGa_U1UE": "Fukushima",
    "any_youtube": "YouTube"
}


class AtomCam:

    ATOM_CAM_RTSP = "rtsp://{}:8554/unicast".format(ATOM_CAM_IP)

    def __init__(self, video_url=ATOM_CAM_RTSP, output=None, end_time="0600",
                 clock=False, mask=None, minLineLength=30, opencl=False):
        self._running = False
        # video device url or movie file path
        self.capture = None
        self.source = None
        self.opencl = opencl

        # 入力ソースの判定
        if "youtube" in video_url:
            # YouTube(マウナケア、木曽、福島、etc)
            self.source = "YouTube"
            for source in YouTube.keys():
                if source in video_url:
                    self.source = YouTube[source]
        else:
            self.source = "ATOMCam"

        self.url = video_url

        self.connect()
        # opencv-python 4.6.0.66 のバグで大きな値(9000)が返ることがあるので対策。
        self.FPS = min(int(self.capture.get(cv2.CAP_PROP_FPS)), 60)
        self.HEIGHT = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WIDTH = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # 出力先ディレクトリ
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = Path('.')
        self.output_dir = output_dir

        # MP4ファイル再生の場合を区別する。
        self.mp4 = Path(video_url).suffix == '.mp4'

        # 終了時刻を設定する。
        now = datetime.now()
        t = datetime.strptime(end_time, "%H%M")
        self.end_time = datetime(
            now.year, now.month, now.day, t.hour, t.minute)
        if now > self.end_time:
            self.end_time = self.end_time + datetime.timedelta(hours=24)

        print("# scheduled end_time = ", self.end_time)
        self.now = now

        if self.source == "ATOMCam" and clock:
            # 内蔵時計のチェック
            check_clock()

        if mask:
            # マスク画像指定の場合
            self.mask = cv2.imread(mask)
        else:
            # 時刻表示部分のマスクを作成
            if self.opencl:
                zero = cv2.UMat((1080, 1920), cv2.CV_8UC3)
            else:
                zero = np.zeros((1080, 1920, 3), np.uint8)

            if self.source == "Subaru":
                # mask SUBRU/Mauna-Kea timestamp
                self.mask = cv2.rectangle(
                    zero, (1660, 980), (1920, 1080), (255, 255, 255), -1)
            elif self.source == "YouTube":
                # no mask
                self.mask = None
            else:
                # mask ATOM Cam timestamp
                self.mask = cv2.rectangle(
                    zero, (1390, 1010), (1920, 1080), (255, 255, 255), -1)

        self.min_length = minLineLength
        self.image_queue = queue.Queue(maxsize=200)

    def __del__(self):
        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        print("# {} stop".format(obs_time))

        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

    def connect(self):
        if self.capture:
            self.capture.release()

        if self.source in YouTube.values():
            # YouTubeからのストリーミング入力
            video = pafy.new(self.url)
            best = video.getbest(preftype="mp4")
            url = best.url
        else:
            url = self.url

        self.capture = cv2.VideoCapture(url)

    def stop(self):
        # thread を止める
        self._running = False

    def queue_streaming(self):
        """RTSP読み込みをthreadで行い、queueにデータを流し込む。
        """
        print("# threading version started.")
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._running = True
        while(True):
            try:
                ret, frame = self.capture.read()
                if self.opencl:
                    frame = cv2.UMat(frame)
                if ret:
                    # self.image_queue.put_nowait(frame)
                    now = datetime.now()
                    self.image_queue.put((now, frame))
                    if self.mp4:
                        current_pos = int(self.capture.get(
                            cv2.CAP_PROP_POS_FRAMES))
                        if current_pos >= frame_count:
                            break
                else:
                    self.connect()
                    datetime.time.sleep(5)
                    continue

                if self._running is False:
                    break
            except Exception as e:
                print(type(e), file=sys.stderr)
                print(e, file=sys.stderr)
                continue

    def dequeue_streaming(self, exposure=1, no_window=False):
        """queueからデータを読み出し流星検知、描画を行う。
        """
        num_frames = int(self.FPS * exposure)

        while True:
            img_list = []
            for n in range(num_frames):
                (t, frame) = self.image_queue.get()
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == 'q':
                    self._running = False
                    return

                if self.mp4 and self.image_queue.empty():
                    self._running = False
                    return

                # exposure time を超えたら終了
                if len(img_list) == 0:
                    t0 = t
                    img_list.append(frame)
                else:
                    dt = t - t0
                    if dt.seconds < exposure:
                        img_list.append(frame)
                    else:
                        break

            if len(img_list) > 2:
                self.composite_img = brightest(img_list)
                if not no_window:
                    cv2.imshow('{}'.format(self.source), self.composite_img)
                self.detect_meteor(img_list)

            # ストリーミングの場合、終了時刻を過ぎたなら終了。
            now = datetime.now()
            if not self.mp4 and now > self.end_time:
                print("# end of observation at ", now)
                self._running = False
                return

    def detect_meteor(self, img_list):
        """img_listで与えられた画像のリストから流星(移動天体)を検出する。
        """
        now = datetime.now()
        obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)

        if len(img_list) > 2:
            # 差分間で比較明合成を取るために最低3フレームが必要。
            # 画像のコンポジット(単純スタック)
            diff_img = brightest(diff(img_list, self.mask))
            try:
                # if True:
                if now.hour != self.now.hour:
                    # 毎時空の様子を記録する。
                    filename = "sky-{:04}{:02}{:02}{:02}{:02}{:02}".format(
                        now.year, now.month, now.day, now.hour, now.minute, now.second)
                    path_name = str(Path(self.output_dir, filename + ".jpg"))
                    mean_img = average(img_list, self.opencl)
                    # cv2.imwrite(path_name, self.composite_img)
                    cv2.imwrite(path_name, mean_img)
                    self.now = now

                detected = detect(diff_img, self.min_length)
                if detected is not None:
                    '''
                    for meteor_candidate in detected:
                        print('{} {} A possible meteor was detected.'.format(obs_time, meteor_candidate))
                    '''
                    print('{} A possible meteor was detected.'.format(obs_time))
                    filename = "{:04}{:02}{:02}{:02}{:02}{:02}".format(
                        now.year, now.month, now.day, now.hour, now.minute, now.second)
                    path_name = str(Path(self.output_dir, filename + ".jpg"))
                    cv2.imwrite(path_name, self.composite_img)

                    # 検出した動画を保存する。
                    movie_file = str(
                        Path(self.output_dir, "movie-" + filename + ".mp4"))
                    self.save_movie(img_list, movie_file)
            except Exception as e:
                print(traceback.format_exc())
                # print(e, file=sys.stderr)

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


def streaming_thread(args):
    """
    RTSPストリーミング、及び動画ファイルからの流星の検出(スレッド版)
    """
    if args.url:
        # URL指定の場合。
        url = args.url
    else:
        # defaultはATOMCamのURL(atomcam_tools版)とする。
        if args.atomcam_tools:
            # atomcam_toolsのRTSPを使う場合。
            url = f"rtsp://{ATOM_CAM_IP}:8554/unicast"
        else:
            # メーカ公式のRTSPを使う場合
            url = f"rtsp://6199:4003@{ATOM_CAM_IP}/live"

    # print(url)
    atom = AtomCam(url, args.output, args.to, args.clock,
                   args.mask, args.min_length)
    if not atom.capture.isOpened():
        return

    now = datetime.now()
    obs_time = "{:04}/{:02}/{:02} {:02}:{:02}:{:02}".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    print("# {} start".format(obs_time))

    # スレッド版の流星検出
    t_in = threading.Thread(target=atom.queue_streaming)
    t_in.start()

    try:
        atom.dequeue_streaming(args.exposure, args.no_window)
    except KeyboardInterrupt:
        atom.stop()

    t_in.join()
    return
