#!/usr/bin/env bash
# Reset and rebind Axelera Metis PCIe endpoint on Rockchip RK3588 platforms

set -e

echo "=== Axelera Metis PCIe Recovery Tool ==="

# 1. Kill any processes holding /dev/metis* or /dev/axl*
echo "[1/5] Terminating any processes locking Metis device nodes..."
sudo killall -9 axtrace 2>/dev/null || true

# 2. Locate the Axelera PCIe device (Vendor ID 1f9d)
AXL_PCI=""
for dev in /sys/bus/pci/devices/*; do
    if [ -f "$dev/vendor" ] && [ "$(cat "$dev/vendor" 2>/dev/null)" = "0x1f9d" ]; then
        AXL_PCI=$(basename "$dev")
        break
    fi
done

if [ -z "$AXL_PCI" ]; then
    AXL_PCI=$(lspci -d 1f9d: -D 2>/dev/null | awk '{print $1}' | head -n1)
fi

if [ -n "$AXL_PCI" ]; then
    echo "[2/6] Found Axelera PCIe device at: $AXL_PCI"
    if [ -f "/sys/bus/pci/devices/$AXL_PCI/remove" ]; then
        echo "      Removing $AXL_PCI from PCIe bus..."
        echo 1 | sudo tee "/sys/bus/pci/devices/$AXL_PCI/remove" >/dev/null || true
        sleep 1
    fi
else
    echo "[2/6] No Axelera device currently listed in sysfs/lspci (will perform bus rescan)."
fi

# 3. Rescan PCIe bus
echo "[3/6] Rescanning PCIe bus..."
echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null
sleep 1

# 4. Clean up any invalid modprobe.d files and reload kernel driver module
echo "[4/6] Reloading driver module..."
sudo rm -f /etc/modprobe.d/axl.conf /etc/modprobe.d/metis.conf 2>/dev/null || true

sudo rmmod axl 2>/dev/null || sudo rmmod axl_aipu 2>/dev/null || sudo rmmod metis 2>/dev/null || true
sleep 1

# Try loading via modprobe or direct insmod
MODULE_LOADED=0
for mod in metis axl; do
    if sudo modprobe "$mod" single_msi=1 dma_poll=0 irq_timeout=30 dma_timeout=10; then
        echo "      Successfully loaded $mod module via modprobe"
        MODULE_LOADED=1
        break
    fi
done

if [ "$MODULE_LOADED" -eq 0 ]; then
    # Look for driver module (.ko or .ko.xz / .ko.zst) across /lib/modules
    KO_FILE=$(sudo find /lib/modules/ -name "metis.ko*" -o -name "axl*.ko*" 2>/dev/null | head -n 1)
    if [ -n "$KO_FILE" ] && [ -f "$KO_FILE" ]; then
        echo "      Loading module directly via insmod: $KO_FILE"
        sudo insmod "$KO_FILE" single_msi=1 dma_poll=0 irq_timeout=30 dma_timeout=10 || true
    else
        echo "      [WARNING] Could not find metis.ko or axl.ko in /lib/modules/"
    fi
fi
sleep 1

# If device is not bound to driver after module load, bind it explicitly
if [ -n "$AXL_PCI" ]; then
    DRIVER_PATH="/sys/bus/pci/devices/$AXL_PCI/driver"
    if [ ! -e "$DRIVER_PATH" ]; then
        for drv in metis axl; do
            if [ -d "/sys/bus/pci/drivers/$drv" ]; then
                echo "      Binding $AXL_PCI to $drv driver..."
                echo "$AXL_PCI" | sudo tee "/sys/bus/pci/drivers/$drv/bind" >/dev/null 2>&1 || true
                break
            fi
        done
    fi
fi

# 5. Verify MSI vector configuration
echo "[5/6] Verifying device status and MSI allocation..."
sleep 1

# Re-discover AXL_PCI after rescan
AXL_PCI=""
for dev in /sys/bus/pci/devices/*; do
    if [ -f "$dev/vendor" ] && [ "$(cat "$dev/vendor" 2>/dev/null)" = "0x1f9d" ]; then
        AXL_PCI=$(basename "$dev")
        break
    fi
done

NUM_MSI=0
if [ -n "$AXL_PCI" ] && [ -d "/sys/bus/pci/devices/$AXL_PCI/msi_irqs" ]; then
    NUM_MSI=$(ls -1 "/sys/bus/pci/devices/$AXL_PCI/msi_irqs" 2>/dev/null | wc -l)
fi

# Also check /proc/interrupts for allocated lines
IRQ_LINES=$(grep -iE "metis|axl" /proc/interrupts 2>/dev/null | wc -l)

# Check module parameter
MOD_MSI=$(cat /sys/module/metis/parameters/single_msi 2>/dev/null || cat /sys/module/axl/parameters/single_msi 2>/dev/null || echo "unknown")
MOD_POLL=$(cat /sys/module/metis/parameters/dma_poll 2>/dev/null || cat /sys/module/axl/parameters/dma_poll 2>/dev/null || echo "unknown")

echo "      PCIe Device:      ${AXL_PCI:-unknown}"
echo "      single_msi:       $MOD_MSI (dma_poll=$MOD_POLL)"
if [ "$NUM_MSI" -gt 0 ]; then
    echo "      sysfs MSI IRQs:   $NUM_MSI vector(s)"
fi
echo "      /proc/interrupts: $IRQ_LINES active IRQ line(s)"

if [ "$NUM_MSI" -eq 1 ] || [ "$IRQ_LINES" -eq 1 ] || [ "$MOD_MSI" = "1" ]; then
    echo "      [SUCCESS] Metis driver active with single_msi=1 protection!"
else
    echo "      [WARNING] Detected $NUM_MSI MSI vectors / $IRQ_LINES IRQ lines."
fi

# Print dmesg driver status
echo "      Recent driver kernel log:"
dmesg 2>/dev/null | grep -iE "metis|axl" | tail -n 3 | sed 's/^/        /' || true

ls -l /dev/metis* /dev/axl* 2>/dev/null || true

# 6. Reset Metis AIPU firmware
echo "[6/6] Reloading Metis AIPU firmware..."
AXDEV=""
for cand in \
    "$(command -v axdevice 2>/dev/null)" \
    "${SUDO_USER:+/home/$SUDO_USER/voyager-sdk/venv/bin/axdevice}" \
    "/home/ubuntu/voyager-sdk/venv/bin/axdevice" \
    "$HOME/voyager-sdk/venv/bin/axdevice" \
    "/usr/local/bin/axdevice" \
    "/usr/bin/axdevice"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then
        AXDEV="$cand"
        break
    fi
done

if [ -n "$AXDEV" ]; then
    echo "      Found axdevice at: $AXDEV"
    "$AXDEV" --reload-firmware || true
else
    echo "      axdevice not found in standard paths, skipping firmware reload."
fi

echo "=== Recovery Complete ==="
