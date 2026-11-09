"""Run tests, saving full output, environment and actual exit code as JSON."""
import argparse
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("reports/validation"))
    parser.add_argument("--pytest",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-m","pytest","-q"] if args.pytest else [sys.executable,"-m","unittest","discover","-s","tests","-v"]
    start=time.perf_counter();result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace")
    (args.out/"tests.log").write_text(result.stdout+result.stderr,encoding="utf-8")
    versions={}
    for name in ["numpy","scipy","pandas","scikit-learn","Pillow","openpyxl","PyYAML","torch","pytest"]:
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:versions[name]="missing"
    payload=dict(command=command,python=platform.python_version(),versions=versions,exit_code=result.returncode,elapsed_seconds=time.perf_counter()-start,scope="CPU synthetic/unit tests; no manuscript-replication claim")
    (args.out/"environment.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(result.stdout+result.stderr)
    print(json.dumps(payload,indent=2))
    return result.returncode

if __name__=="__main__":raise SystemExit(main())
