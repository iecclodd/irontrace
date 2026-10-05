"""Private local SQLite persistence. Only explicitly public payloads leave the API."""
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import threading

class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, payload TEXT NOT NULL, row_index INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS purchases(session_id TEXT, token TEXT, group_id TEXT, PRIMARY KEY(session_id, token));
            CREATE TABLE IF NOT EXISTS events(session_id TEXT, sequence INTEGER, payload TEXT, PRIMARY KEY(session_id,sequence));
        ''')
        self.connection.commit()
        self.lock = threading.RLock()

    @contextmanager
    def transaction(self):
        with self.lock:
            self.connection.execute('BEGIN IMMEDIATE')
            try:
                yield self
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def get(self, sid):
        with self.lock:
            result = self.connection.execute('SELECT payload,row_index FROM sessions WHERE id=?', (sid,)).fetchone()
            if not result:
                raise KeyError(sid)
            return json.loads(result[0]), result[1]

    def save(self, payload, row_index):
        self.connection.execute('INSERT INTO sessions VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
            (payload['id'], json.dumps(payload, allow_nan=False), int(row_index)))

    def purchase(self, sid, token):
        row = self.connection.execute('SELECT group_id FROM purchases WHERE session_id=? AND token=?', (sid, token)).fetchone()
        return row[0] if row else None

    def record_purchase(self, sid, token, group_id):
        self.connection.execute('INSERT INTO purchases VALUES(?,?,?)', (sid, token, group_id))

    def append_event(self, sid, event):
        seq = self.connection.execute('SELECT COUNT(*) FROM events WHERE session_id=?',(sid,)).fetchone()[0]
        self.connection.execute('INSERT INTO events VALUES(?,?,?)',(sid,seq,json.dumps(event,allow_nan=False)))

    def events(self, sid):
        with self.lock:
            return [json.loads(r[0]) for r in self.connection.execute('SELECT payload FROM events WHERE session_id=? ORDER BY sequence',(sid,))]
