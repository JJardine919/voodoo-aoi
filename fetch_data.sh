#!/usr/bin/env bash
# Download the public datasets for the demos into ./data/ .
# All sources are open-access. Idempotent: re-running skips files already present.
set -euo pipefail

DATA="${VOODOO_DATA:-data}"
mkdir -p "$DATA/mri" "$DATA/ligo" "$DATA/battery"

dl() { # dl <url> <dest>
  if [ -s "$2" ]; then echo "  have $2"; return; fi
  echo "  fetch $2"
  curl -fL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

echo "[1/3] MRI  — OpenNeuro ds000102 T1w (6 subjects)"
for n in 01 02 03 04 05 06; do
  dl "https://s3.amazonaws.com/openneuro.org/ds000102/sub-${n}/anat/sub-${n}_T1w.nii.gz" \
     "$DATA/mri/sub-${n}_T1w.nii.gz"
done

echo "[2/3] LIGO — Gravity Spy O1 H1 glitch catalog (Zenodo 5649212)"
dl "https://zenodo.org/records/5649212/files/H1_O1.csv?download=1" \
   "$DATA/ligo/H1_O1.csv"
# (strain windows themselves are fetched live from GWOSC by demos/ligo_glitches.py)

echo "[3/3] Battery — NASA PCoE Li-ion aging set (B0005/6/7/18)"
if [ -s "$DATA/battery/B0005.mat" ] && [ -s "$DATA/battery/B0018.mat" ]; then
  echo "  have battery .mat files"
else
  ZIP="$DATA/battery/nasa_battery.zip"
  dl "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip" "$ZIP"
  echo "  unpacking (nested zips)..."
  TMP="$DATA/battery/_unzip"; rm -rf "$TMP"; mkdir -p "$TMP"
  # The NASA archive is a zip of zips; extract recursively until no .zip remains.
  python3 - "$ZIP" "$TMP" <<'PY'
import zipfile, sys, os
outer, tmp = sys.argv[1], sys.argv[2]
zipfile.ZipFile(outer).extractall(tmp)
while True:
    inner = [os.path.join(r,f) for r,_,fs in os.walk(tmp) for f in fs if f.lower().endswith(".zip")]
    if not inner: break
    for z in inner:
        try: zipfile.ZipFile(z).extractall(os.path.dirname(z))
        except zipfile.BadZipFile: pass
        os.remove(z)
PY
  for cell in B0005 B0006 B0007 B0018; do
    f=$(find "$TMP" -name "${cell}.mat" | head -1)
    [ -n "$f" ] || { echo "ERROR: ${cell}.mat not found in NASA zip" >&2; exit 1; }
    cp "$f" "$DATA/battery/${cell}.mat"
  done
  rm -rf "$TMP" "$ZIP"
fi

echo
echo "Done. Data is under ./$DATA/ . Now run:"
echo "  python demos/mri_qc.py"
echo "  python demos/ligo_glitches.py"
echo "  python demos/battery_predict.py"
