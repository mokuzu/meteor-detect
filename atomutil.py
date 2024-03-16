#!/usr/bin/env python

import datetime
from pathlib import Path
import argparse
import telnetlib
import cv2
from atomcam import ATOM_CAM_IP

from atomcam_videofile import DetectMeteor

# atomcam_toolsでのデフォルトのユーザアカウントなので、自分の環境に合わせて変更してください。
ATOM_CAM_USER = "root"
ATOM_CAM_PASS = "atomcam2"

class AtomTelnet():
    '''
    ATOM Camにtelnet接続し、コマンドを実行するクラス
    '''

    def __init__(self, ip_address=ATOM_CAM_IP):
        """AtomTelnetのコンストラクタ

        Args:
          ip_address: Telnet接続先のIPアドレス
        """
        self.tn = telnetlib.Telnet(ip_address)
        self.tn.read_until(b"login: ")
        self.tn.write(ATOM_CAM_USER.encode('ascii') + b"\n")
        self.tn.read_until(b"Password: ")
        self.tn.write(ATOM_CAM_PASS.encode('ascii') + b"\n")

        self.tn.read_until(b"# ")

    def exec(self, command):
        """Telnet経由でコマンドを実行する。

        Args:
          command : 実行するコマンド(ex. "ls")

        Returns:
          コマンド実行結果文字列。1行のみ。
        """
        self.tn.write(command.encode('utf-8') + b'\n')
        ret = self.tn.read_until(b"# ").decode('utf-8').split("\r\n")[1]
        return ret

    def exit(self):
        self.tn.write("exit".encode('utf-8') + b"\n")

    def __del__(self):
        self.exit()


def set_clock():
    """ATOM Camのクロックとホスト側のクロックに合わせる。
    """
    tn = AtomTelnet()
    # utc_now = datetime.now(timezone.utc)
    jst_now = datetime.now()
    set_command = 'date -s "{}"'.format(jst_now.strftime("%Y-%m-%d %H:%M:%S"))
    print(set_command)
    tn.exec(set_command)


def check_clock():
    """ATOM Camのクロックとホスト側のクロックの比較。
    """
    tn = AtomTelnet()
    atom_date = tn.exec('date')
    '''
    utc_now = datetime.now(timezone.utc)
    atom_now = datetime.strptime(atom_date, "%a %b %d %H:%M:%S %Z %Y")
    atom_now = atom_now.replace(tzinfo=timezone.utc)
    '''
    jst_now = datetime.now()
    atom_now = datetime.strptime(atom_date, "%a %b %d %H:%M:%S %Z %Y")

    dt = atom_now - jst_now
    if dt.days < 0:
        delta = -(86400.0 - (dt.seconds + dt.microseconds/1e6))
    else:
        delta = dt.seconds + dt.microseconds/1e6

    print("# ATOM Cam =", atom_now)
    print("# HOST PC  =", jst_now)
    print("# ATOM Cam - Host PC = {:.3f} sec".format(delta))


def make_ftpcmd(meteor_list, directory):
    '''
    検出されたログから画像をダウンロードするFTPコマンドを生成する。
    '''
    wget = "wget -nc -r -nv -nH --cut-dirs=3"
    if directory:
        wget += " -P {}".format(directory)

    with open(meteor_list, "r") as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue

            # 検出時刻から動画ファイル名を生成する。
            (date, time) = line.split()[0:2]
            hh, mm, ss = time.split(':')
            date_dir = ''.join(date.split('/'))
            mp4_file = "{}/{}/{}.mp4".format(date_dir, hh, mm)
            url = "ftp://{}:{}@{}/media/mmc/record/{}".format(
                ATOM_CAM_USER, ATOM_CAM_PASS, ATOM_CAM_IP, mp4_file
            )
            print("{} {}".format(wget, url))


def detect_meteors(meteor_list):
    '''
    検出された流星リストから再検出(修正中)
    '''
    with open(meteor_list, "r") as f:
        prev_file = None
        for line in f.readlines():
            if line.startswith('#'):
                continue

            (date, time) = line.split()[0:2]
            date_dir = ''.join(date.split('/'))

            hh, mm, ss = time.split(':')

            file_path = Path(date_dir, hh, "{}.mp4".format(mm))

            if file_path != prev_file:
                print(file_path)
                detecter = DetectMeteor(str(file_path))
                detecter.meteor(2)

                prev_file = file_path


def make_movie(meteor_list, output="movie.mp4", fps=1.0):
    '''
    検出された流星リストから動画作成(未完成)

    Args:
      meteor_list: 検出された流星のログファイル
      outout: 出力動画ファイル名
    '''
    data_dir = Path(meteor_list).parents[0]
    date_dir = Path(meteor_list).stem

    # とりあえずATOM Camサイズ
    size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    print(f'FPS={fps}')
    video = cv2.VideoWriter(output, fourcc, fps, size)

    with open(meteor_list, "r") as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue

            (date, time) = line.split()[0:2]
            date_str = ''.join(date.split('/'))

            hh, mm, ss = time.split(':')
            filename = "{}{}{}{}.jpg".format(date_str, hh, mm, ss)
            file_path = str(Path(data_dir, date_dir, filename))

            print(file_path)
            try:
                img = cv2.imread(file_path)
                video.write(img)
            except Exception as e:
                print(str(e))

        video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('meteors', nargs='?', help="List of detected meteors (text file)")
    parser.add_argument('-f', '--ftp', action='store_true', help='FTPコマンド作成')
    parser.add_argument('-d', '--directory', default=None, help='FTPコマンド取得先ディレクトリ名')
    parser.add_argument('-m', '--movie', action='store_true', help='FTPコマンド作成')
    parser.add_argument('-o', '--output', default='movie.mp4', help='動画ファイル名(.mp4)')
    parser.add_argument('-c', '--clock', action='store_true', help='ATOM Camの時計のチェック')
    parser.add_argument('-s', '--set_clock', action='store_true', help='ATOM Camの時計をホスト側に合わせる')
    parser.add_argument('-F', '--fps', default=1, type=int, help='動画生成時のFPS')

    args = parser.parse_args()

    # print("# {}".format(args.meteors))

    if args.ftp:
        make_ftpcmd(args.meteors, args.directory)
    elif args.movie:
        make_movie(args.meteors, args.output, args.fps)
        #make_movie(args.meteors, args.output)
    elif args.clock:
        check_clock()
    elif args.set_clock:
        set_clock()
    else:
        detect_meteors(args.meteors)
