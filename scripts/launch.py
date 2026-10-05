"""Start one local numerical worker and serve the built console."""
from pathlib import Path
import argparse
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    os.chdir(ROOT)
    import uvicorn
    print(f'NextCheck: http://127.0.0.1:{args.port} (one model worker; local binding)')
    uvicorn.run('nextcheck.api:app',host='127.0.0.1',port=args.port,workers=1)

if __name__=='__main__': main()
