import os
import time
import urllib.request


def download_to_stream(url, stream, size_in_mb):
    # download to stream file by 10MB per chunk

    cur = 1
    begin = time.time()

    def progress():
        nonlocal cur
        mb_read = cur * 10
        pct_read = 100 * mb_read / size_in_mb
        mb_s = mb_read / (time.time() - begin)
        rem_s = (size_in_mb - mb_read) / mb_s
        print(
            f"\r{pct_read:.2f}%, {mb_s:.1f} MB/s, eta {rem_s:.0f} s. ",
            end="", flush=True
        )
        cur += 1

    with urllib.request.urlopen(url) as fi:
        assert fi.status == 200
        chunk = fi.read(10*1024**2)
        while chunk:
            stream.write(chunk)
            progress()
            chunk = fi.read(10*1024**2)
        print()


def download_debug(fname, stream, size_in_mb):
    # stream a local file for debugging purposes

    cur = 1
    begin = time.time()

    def progress():
        nonlocal cur
        mb_read = cur * 10
        pct_read = 100 * mb_read / size_in_mb
        mb_s = mb_read / (time.time() - begin)
        rem_s = (size_in_mb - mb_read) / mb_s
        print(
            f"\r{pct_read:.2f}%, {mb_s:.1f} MB/s, eta {rem_s:.0f} s. ",
            end="", flush=True
        )
        cur += 1

    with open(fname, "rb") as fp:
        chunk = fp.read(10*1024**2)
        while chunk:
            stream.write(chunk)
            progress()
            chunk = fp.read(10*1024**2)
        print()


def download(url, fname, stream, size_in_mb):
    if os.path.exists(fname):
        download_debug(fname, stream, size_in_mb)
    else:
        download_to_stream(url, stream, size_in_mb)


def make_directories(dataset_name):
    for subset in ("train", "val", "test"):
        os.makedirs(f"{dataset_name}/{subset}/annotations", exist_ok=False)
        os.makedirs(f"{dataset_name}/{subset}/photos", exist_ok=False)


def check_equal_annotations_photos(dataset_name):
    for subset in ("train", "val", "test"):
        an = len(os.listdir(f"{dataset_name}/{subset}/annotations"))
        im = len(os.listdir(f"{dataset_name}/{subset}/photos"))
        assert an == im
        print(f"{subset}: {an}")


def zimg_to_disk(zf, obj, *path):
    out = os.path.join(*path)
    with open(out, "wb") as f:
        f.write(zf.read(obj.filename))
