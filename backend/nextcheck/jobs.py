"""One serialized numerical worker per application process."""
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from uuid import uuid4
from copy import deepcopy

class Jobs:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nextcheck-model')
        self.items = {}
        self.lock = RLock()

    def submit(self, function):
        jid = uuid4().hex
        with self.lock:
            self.items[jid] = dict(id=jid,status='queued',progress=0,error=None,result=None)
        def progress(value):
            with self.lock:
                self.items[jid]['progress'] = value
        def execute():
            with self.lock:
                self.items[jid]['status'] = 'running'
            try:
                result = function(progress)
                with self.lock:
                    self.items[jid].update(status='completed',progress=1,result=result)
            except Exception as exc:
                with self.lock:
                    self.items[jid].update(status='error',error=str(exc)[:1500])
        self.executor.submit(execute)
        return jid

    def get(self, jid):
        with self.lock:
            return deepcopy(self.items[jid])

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

