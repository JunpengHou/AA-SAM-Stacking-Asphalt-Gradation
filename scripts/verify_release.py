"""Read-only publication audit; never stages, deletes or uploads files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

SKIP={".git",".work",".venv","__pycache__",".pytest_cache","results","reports","artifacts","build","dist"}
SUFFIXES={".py",".md",".toml",".yaml",".yml",".txt",".json",".csv"}
PRIVATE={".pt",".pth",".ckpt",".joblib",".xlsx",".xls",".pdf",".docx",".parquet"}


def candidates(root):
    for path in root.rglob("*"):
        relative=path.relative_to(root)
        if any(part in SKIP or part.startswith(".local_deps") or part.endswith(".egg-info") for part in relative.parts):continue
        if relative.parts[0]=="data" and not (path.name.endswith(".example.csv") or path.name=="README.md"):continue
        if path.is_file() and (path.suffix in SUFFIXES or path.name==".gitignore"):yield path


def scan(root):
    root=Path(root).resolve();issues=[];files={}
    private_path=re.compile(r"[A-Za-z]:[\\/](?:Users|image|Python|small_paper|Study)[\\/]",re.I)
    secret=re.compile(r"(?:sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
    for path in candidates(root):
        relative=path.relative_to(root).as_posix();raw=path.read_bytes()
        files[relative]=hashlib.sha256(raw).hexdigest()
        for line_number,line in enumerate(raw.decode("utf-8",errors="replace").splitlines(),1):
            if private_path.search(line):issues.append(dict(kind="machine_specific_path",file=relative,line=line_number))
            if secret.search(line):issues.append(dict(kind="potential_secret",file=relative,line=line_number))
    if not (root/"LICENSE").is_file():issues.append(dict(kind="license_unconfirmed"))
    git=subprocess.run(["git","ls-files","-z"],cwd=root,capture_output=True)
    if git.returncode:issues.append(dict(kind="git_index_unavailable"))
    else:
        for name in git.stdout.decode("utf-8").split("\0"):
            if not name:continue
            path=Path(name)
            if path.suffix.lower() in PRIVATE or any(p in {".work","results","reports","artifacts"} or p.startswith(".local_deps") for p in path.parts):
                issues.append(dict(kind="private_artifact_in_index",file=name))
    return dict(release_ready=not issues,issues=issues,public_source_sha256=files,
                limitations="Static candidate scan and Git index only; legal clearance and actual provenance require human evidence")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args();result=scan(Path(__file__).resolve().parents[1])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(dict(release_ready=result["release_ready"],n_issues=len(result["issues"]),n_sources=len(result["public_source_sha256"])),indent=2))
    return 0 if result["release_ready"] else 1

if __name__=="__main__":raise SystemExit(main())
