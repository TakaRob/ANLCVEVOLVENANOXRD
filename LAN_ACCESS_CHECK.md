# LAN access check — can this computer bin the data fast?

**Why:** binning each scan reads its full ~283 GB of raw frames once. On the
laptop the micdata link tops out at **~23 MB/s** → **~42 h** to bin the 12 study
scans (203–214). If a computer on the **micdata LAN** can read at ~100+ MB/s, we
bin *there* and download only the small binned `.h5` files (<1 h). This check
tells you which case you're in — on whatever computer you run it.

The data share is `\\micdata\data1` (Windows `Z:`); the project lives under
`isn/2026-1/2026-1-Luo/`.

---

## Quick start (Linux / macOS / WSL)

```bash
git pull                              # get this repo on the other computer
chmod +x check_lan_access.sh
./check_lan_access.sh                 # auto-probes common mount points
# or point it at the data explicitly:
./check_lan_access.sh /path/to/data1/isn/2026-1/2026-1-Luo
```

The script **only reads** (to `/dev/null`) — it writes nothing and needs no
write space. It finds the data, confirms scans 203–214 are present, then reads a
~1.2 GB XRD file and prints the real bandwidth.

### If it can't find the data
Mount the share first, then pass the path. Example SMB (CIFS) mount on Linux:

```bash
sudo mkdir -p /mnt/micdata
sudo mount -t cifs //micdata.xray.aps.anl.gov/data1 /mnt/micdata \
     -o username=YOURUSER,domain=ANL,uid=$(id -u),ro,vers=3.0
./check_lan_access.sh /mnt/micdata/isn/2026-1/2026-1-Luo
```
(Adjust `domain=` / `vers=` if your site differs. `ro` = read-only, safe.)

---

## Reading the result

The script ends with a line like:

```
MEASURED READ SPEED:  137.4 MB/s     (laptop WAN baseline ≈ 23 MB/s)
Est. time to bin all 12 scans here: ~6.9 h   (laptop ≈ 42 h)
```

| Speed | Meaning | Action |
|-------|---------|--------|
| **≥ 80 MB/s** | ✅ Fast LAN | **Bin here**, download small binned `.h5` |
| **~46–80 MB/s** | 🟡 Better than laptop | Binning here still beats pulling raw |
| **≤ ~23 MB/s** | ❌ No faster | Not closer to the data; find a LAN box or accept ~42 h |

**Report the `MEASURED READ SPEED` line back** and we'll pick the path.

---

## Windows PowerShell equivalent (no repo needed)

If the "other computer" is Windows with `Z:` mapped, paste this into PowerShell:

```powershell
$src = (Get-ChildItem 'Z:\isn\2026-1\2026-1-Luo\Raw\Scan_0203\XRD\*.h5' |
        Select-Object -First 1).FullName
$sz  = (Get-Item $src).Length / 1MB
$t   = Measure-Command { Copy-Item $src "$env:TEMP\bw.h5" -Force }
'{0:N1} MB/s  ({1:N0} MB in {2:N1}s)' -f ($sz/$t.TotalSeconds), $sz, $t.TotalSeconds
Remove-Item "$env:TEMP\bw.h5" -Force
```

≥80 MB/s → bin on that machine; ~23 MB/s → it's the same WAN link as the laptop.

---

## Context

Full study + pipeline plan: [`ROCKING_STUDY_203-214.md`](ROCKING_STUDY_203-214.md).
The binning-location decision (here, gated by this check) is the single biggest
lever on the whole study timeline.
