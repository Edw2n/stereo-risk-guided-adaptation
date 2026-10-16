import io, re, time, requests

class RangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if self.pos >= end:
            return b''
        if end - self.pos > 100_000_000:
            raise ValueError('Unexpected range exceeds 100 MB')
        expected = f'bytes {self.pos}-{end-1}/{self.size}'
        for attempt in range(4):
            try:
                with self.session.get(self.url, headers={'Range': f'bytes={self.pos}-{end-1}'},
                                      timeout=(10, 20), stream=True) as r:
                    if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                        raise RuntimeError(f'Range response invalid: {r.status_code}, {r.headers.get("Content-Range")}')
                    data = r.content
                if len(data) != end - self.pos:
                    raise IOError('Truncated range')
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        self.pos = end
        self.transferred += len(data)
        return data

