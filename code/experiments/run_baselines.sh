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
printf '%s\n' "$PQC_SUDO_PASSWORD" | sudo -S -v
mkdir -p "${out_dir}"
cd "${out_dir}"

# 1. dm-crypt setup
echo "[*] Setting up dm-crypt..."
fallocate -l 5G dm_crypt.img
LOOP_DM=$(printf '%s\n' "$PQC_SUDO_PASSWORD" | sudo -S losetup -f --show dm_crypt.img)
echo -n "testpass" | sudo cryptsetup luksFormat --batch-mode $LOOP_DM -
echo -n "testpass" | sudo cryptsetup open $LOOP_DM dmcrypt_test -
sudo mkfs.ext4 /dev/mapper/dmcrypt_test
mkdir -p mnt_dm
sudo mount /dev/mapper/dmcrypt_test mnt_dm
sudo chown -R thor:thor mnt_dm

# 2. fscrypt setup
echo "[*] Setting up fscrypt..."
fallocate -l 5G fscrypt.img
LOOP_FS=$(sudo losetup -f --show fscrypt.img)
sudo mkfs.ext4 -O encrypt $LOOP_FS
sudo tune2fs -O encrypt $LOOP_FS || true
mkdir -p mnt_fs
sudo mount $LOOP_FS mnt_fs
sudo chown -R thor:thor mnt_fs
sudo fscrypt setup --quiet || true
sudo fscrypt setup mnt_fs --quiet || true
echo -n "testpass" | sudo fscrypt encrypt mnt_fs --source=custom_passphrase --quiet || true

# 3. FIO Benchmarks
echo "[*] Running FIO on dm-crypt..."
sudo fio --name=seqwrite_dm --ioengine=libaio --rw=write --bs=1M --size=2G --numjobs=1 --iodepth=16 --directory=mnt_dm --output-format=json > dm_crypt_fio.json

echo "[*] Running FIO on fscrypt..."
sudo fio --name=seqwrite_fs --ioengine=libaio --rw=write --bs=1M --size=2G --numjobs=1 --iodepth=16 --directory=mnt_fs/encrypted_test --output-format=json > fscrypt_fio.json || sudo fio --name=seqwrite_fs --ioengine=libaio --rw=write --bs=1M --size=2G --numjobs=1 --iodepth=16 --directory=mnt_fs --output-format=json > fscrypt_fio.json

# Cleanup
echo "[*] Cleaning up..."
sudo umount mnt_dm || true
sudo cryptsetup close dmcrypt_test || true
sudo losetup -d $LOOP_DM || true

sudo umount mnt_fs || true
sudo losetup -d $LOOP_FS || true

echo "[+] Baseline evaluation complete."
