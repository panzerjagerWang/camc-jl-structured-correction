"""Download public assets and pinned upstream files; validate each SHA-256."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess, urllib.request

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def fetch(url, path, expected):
    if path.exists() and digest(path) == expected:
        print("Verified", path.relative_to(ROOT))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "CAMC-JL-download/2.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if digest(part) != expected:
            raise ValueError("SHA-256 mismatch: " + path.name)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    print("Downloaded and verified", path.relative_to(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="Actual GitHub OWNER/REPOSITORY")
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads((ROOT / "WEIGHTS.json").read_text())
    if args.repo:
        repo = args.repo
    else:
        publication = json.loads((ROOT / "PUBLICATION.json").read_text())
        remote = publication.get("repository_url")
        if not remote:
            remote = subprocess.check_output(
                ["git", "-C", str(ROOT), "remote", "get-url", "origin"], text=True).strip()
        match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?/?", remote)
        if not match:
            parser.error("Cannot infer repository; pass --repo OWNER/REPOSITORY")
        repo = match.group(1)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        parser.error("Invalid OWNER/REPOSITORY")
    for asset in spec["assets"]:
        if args.frontend_only and "/adapters/" in asset["destination"]:
            continue
        url = f"https://github.com/{repo}/releases/download/{spec['release_tag']}/{asset['name']}"
        fetch(url, ROOT / asset["destination"], asset["sha256"])
    if not args.frontend_only:
        manifest = json.loads((ROOT / "model/manifest.json").read_text())
        for name, expected in manifest["files_sha256"].items():
            if not name.startswith("base/"):
                continue
            filename = name.removeprefix("base/")
            url = f"https://huggingface.co/{spec['base_repository']}/resolve/{spec['base_revision']}/{filename}"
            fetch(url, ROOT / "model" / name, expected)

if __name__ == "__main__":
    main()
