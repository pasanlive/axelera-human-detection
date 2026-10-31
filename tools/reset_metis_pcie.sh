#!/usr/bin/env bash
# Reset and rebind Axelera Metis PCIe endpoint on Rockchip RK3588 platforms

set -e

echo "=== Axelera Metis PCIe Recovery Tool ==="

# 1. Kill any processes holding /dev/metis* or /dev/axl*
echo "[1/5] Terminating any processes locking Metis device nodes..."
sudo killall -9 axtrace 2>/dev/null || true

# 2. Locate the Axelera PCIe device (Vendor ID 1f9d)
AXL_PCI=$(lspci -d 1f9d: -D 2>/dev/null | awk '{print $1}' | head -n1)

if [ -n "$AXL_PCI" ]; then
    echo "[2/5] Found Axelera PCIe device at: $AXL_PCI"
    if [ -f "/sys/bus/pci/devices/$AXL_PCI/remove" ]; then
        echo "      Removing $AXL_PCI from PCIe bus..."
        echo 1 | sudo tee "/sys/bus/pci/devices/$AXL_PCI/remove" >/dev/null || true
        sleep 1
    fi
else
    echo "[2/5] No Axelera device currently listed in lspci (may need bus rescan)."
fi

# 3. Rescan PCIe bus
echo "[3/5] Rescanning PCIe bus..."
echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null
sleep 1

# 4. Reload kernel driver module
echo "[4/5] Reloading driver module (axl/metis) with single_msi=1 and dma_poll=1..."
sudo rmmod axl 2>/dev/null || sudo rmmod metis 2>/dev/null || true
sleep 1

if modinfo axl >/dev/null 2>&1; then
    sudo modprobe axl single_msi=1 dma_poll=1 irq_timeout=30 dma_timeout=10
elif modinfo metis >/dev/null 2>&1; then
    sudo modprobe metis single_msi=1 dma_poll=1 irq_timeout=30 dma_timeout=10
fi

# 5. Verify status
echo "[5/5] Verifying device status..."
sleep 1
lspci -d 1f9d: -k || true
ls -l /dev/metis* /dev/axl* 2>/dev/null || true

echo "=== Recovery Complete ==="
