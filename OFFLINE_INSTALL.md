# Offline install — RHEL 9.6 / Python 3.12

Pre-downloaded wheels for the nano-XRD environment, for an **offline RHEL 9.6
(x86-64)** machine running **Python 3.12** (RHEL 9 app-stream `python3.12`).

- Wheels live in `pip_packages_rhel96_py312/` (137 wheels, ~297 MB, all binary —
  nothing compiles on the target).
- Package list: `requirements-offline.txt` (third-party deps only).
- Built from the dev `.venv` (Python 3.14) via `pip download` targeting
  `--python-version 3.12 --platform manylinux_2_34/2_28/2014_x86_64`.

## 1. Get the files onto the offline machine

The offline box reads the mounted drive, so copy the bundle there (or it can read
it straight from the mount). Copy **both** the wheel folder and the requirements
file:

```
pip_packages_rhel96_py312/
requirements-offline.txt
```

## 2. Create a venv with Python 3.12 on the RHEL box

```bash
sudo dnf install -y python3.12        # if not already present (app-stream)
python3.12 -m venv ~/xrd-venv
source ~/xrd-venv/bin/activate
```

## 3. Install everything offline

Point at the drive location where you put the bundle (adjust `DIR`):

```bash
DIR=/path/to/bundle          # dir containing pip_packages_rhel96_py312/ and requirements-offline.txt
pip install --no-index --find-links="$DIR/pip_packages_rhel96_py312" \
    -r "$DIR/requirements-offline.txt"
```

- `--no-index` = never contact PyPI
- `--find-links=<dir>` = install from the local wheel folder only

## 4. Install your own code (xrd_app / xrd_tools)

These are your repo packages (not on PyPI). From the repo checkout on the drive:

```bash
pip install --no-index --no-build-isolation -e /path/to/ANLCVEVOLVENANOXRD
```

(`--no-build-isolation` because there's no internet to fetch build backends; the
wheels above already provide setuptools/wheel.)

## Notes / deviations from the dev env

- **PyQt5** was relaxed from `5.15.11` → pip picked **5.15.7** (newest with a
  py3.12/RHEL wheel). Functionally equivalent for our use.
- **mictools / minepy** were **dropped** — `minepy` is sdist-only (needs a C
  compiler) and is **not imported anywhere in the repo**. If you ever need it,
  install it on a machine with a compiler, or add `--no-binary minepy` plus gcc +
  numpy headers on the target.
- Everything else matches the dev pins exactly.
- If `dnf` is also offline, you'll need `python3.12` from the RHEL install media /
  a local dnf repo — pip can't provide the interpreter itself.

## To refresh this bundle later (on a machine WITH internet)

```bash
cd "<repo>"
.venv/bin/python -m pip download --only-binary=:all: \
  --python-version 3.12 --implementation cp \
  --platform manylinux_2_34_x86_64 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 \
  -r requirements-offline.txt -d pip_packages_rhel96_py312
```
