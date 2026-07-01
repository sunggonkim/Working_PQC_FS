#!/bin/bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
root_dir="$(cd "${script_dir}/../.." && pwd -P)"
out_dir="${root_dir}/artifacts/results/baselines"

echo "=== AEGIS-Q Baseline Evaluation ==="
if [ -z "${PQC_SUDO_PASSWORD:-}" ]; then
    echo "PQC_SUDO_PASSWORD must be set for loop-device and mount setup" >&2
    exit 1
fi
sudo_pw() {
    sudo -n "$@"
}

cleanup() {
    set +e
    [ -d "mnt_dm" ] && sudo_pw umount mnt_dm
    sudo_pw cryptsetup close dmcrypt_test
    [ -n "${LOOP_DM:-}" ] && sudo_pw losetup -d "$LOOP_DM"
}
trap cleanup EXIT

printf '%s\n' "$PQC_SUDO_PASSWORD" | sudo -S -v
mkdir -p "${out_dir}"
cd "${out_dir}"

# Ensure reruns start from a clean state.
sudo_pw umount mnt_dm 2>/dev/null || true
sudo_pw cryptsetup close dmcrypt_test 2>/dev/null || true
rm -f dm_crypt.img dm_crypt_fio.json

# 1. dm-crypt setup
echo "[*] Setting up dm-crypt..."
fallocate -l 5G dm_crypt.img
LOOP_DM=$(sudo_pw losetup -f --show dm_crypt.img)
echo -n "testpass" | sudo_pw cryptsetup luksFormat --batch-mode "$LOOP_DM" -
echo -n "testpass" | sudo_pw cryptsetup open "$LOOP_DM" dmcrypt_test -
sudo_pw mkfs.ext4 /dev/mapper/dmcrypt_test
mkdir -p mnt_dm
sudo_pw mount /dev/mapper/dmcrypt_test mnt_dm
sudo_pw chown -R thor:thor mnt_dm

# 2. FIO Benchmarks
echo "[*] Running FIO on dm-crypt..."
sudo_pw fio --name=seqwrite_dm --ioengine=libaio --rw=write --bs=1M --size=2G --numjobs=1 --iodepth=16 --directory=mnt_dm --output-format=json > dm_crypt_fio.json

echo "[+] Baseline evaluation complete."
