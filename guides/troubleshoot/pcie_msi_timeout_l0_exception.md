# Troubleshooting: Level-Zero C++ Exception (`L0::Exception`), MSI Timeout & Pipeline Stall

## Problem Statement

When starting the Axelera Metis multi-camera pipeline (`python app.py`), the worker process crashes after running for a few seconds (e.g. 7–45s) with a native Level-Zero C++ abort (`SIGABRT`, exit code `-6`), even when `--reload-firmware` succeeds at startup:

```text
[METIS HW] Firmware reloaded: Device 0: metis-0:1:0 4GiB metis-compute-board flver=1.5.0 bcver=7.1 clock=800MHz(0-3:800MHz) mvm=0-3:100%
...
[STATUS] tier=high target=15fps/cam achieved=20.5fps total persons=0 frames=6 lat=2.8ms (p95=192.6ms)

[WATCHDOG ALERT] Inference pipeline stall detected! No frame received for 10.0s (threshold=10.0s).
  Last completed frame: #6 on 'cam_02' (latency: 2.8ms). Checking hardware status...
  (No new kernel dmesg messages)
  /dev/metis-0:1:0 is present in /dev.
...
[libaxldev.c:140] Failed to wait for MSI 0: Connection timed out
[ERROR][waitForQueueBinaryCompletion]: Wait kernel failed with return code -1.
terminate called after throwing an instance of 'L0::Exception'

[SUPERVISOR ALERT] Worker PID 104265 crashed with exit code -6 (SIGABRT) after 43.5s!
[SUPERVISOR ALERT] Diagnosed Reason: Level-Zero C++ exception (L0::Exception) thrown from Metis driver thread
```

On subsequent auto-restarts, it fails even faster (within 12–18s) with:

```text
[ERROR][waitForQueueBinaryCompletion]: Wait kernel failed with return code -1301.
terminate called after throwing an instance of 'L0::Exception'

[SUPERVISOR ALERT] Worker PID 104471 crashed with exit code -6 (SIGABRT) after 18.8s!
[SUPERVISOR ALERT] Diagnosed Reason: Level-Zero wait kernel timeout (-1301): AIPU command queue exceeded wait window
```

---

## Log Signatures & Identification

Look for these diagnostic messages in your terminal and `dmesg`:

### 1. Initial Frame Output Followed by Complete Pipeline Stall
```text
[STATUS] tier=high target=15fps/cam achieved=20.5fps total persons=0 frames=6 lat=2.8ms (p95=192.6ms)
[WATCHDOG ALERT] Inference pipeline stall detected! No frame received for 10.0s (threshold=10.0s).
```

### 2. Low-Level User-Space Driver Timeout
```text
[libaxldev.c:140] Failed to wait for MSI 0: Connection timed out
[ERROR][waitForQueueBinaryCompletion]: Wait kernel failed with return code -1.
terminate called after throwing an instance of 'L0::Exception'
```
or return code `-1301` / `1`:
```text
[ERROR][waitForQueueBinaryCompletion]: Wait kernel failed with return code -1301.
terminate called after throwing an instance of 'L0::Exception'
```

### 3. Kernel Diagnostic Log Entries
```text
[Wed Nov  4 02:20:51 2026] axl 0000:01:00.0: vmsi configured
[Wed Nov  4 02:20:52 2026] axl 0000:01:00.0: IRQ MSI timeout (12 1)
[Wed Nov  4 02:21:32 2026] axl 0000:01:00.0: IRQ MSI timeout (0 30)
```

---

## Root Cause Analysis

1. **Why `--reload-firmware` Alone Does Not Prevent the Crash:**
   `--reload-firmware` re-flashes the on-chip firmware into SRAM and checks device responsiveness over PCIe register space, which succeeds (`flver=1.5.0 bcver=7.1 clock=800MHz`). However, **it does not reset the Rockchip RK3588 host PCIe controller's link state or clear driver ring buffers**.
2. **Interrupt Latency & Dropped MSI Interrupts:**
   Under standard interrupt DMA mode (`dma_poll=0`), the driver waits for a physical hardware MSI interrupt from the AIPU to notify `libaxldev` that an inference binary/queue command has completed.
   On RK3588 SBCs (e.g. Radxa Rock 5B, Orange Pi 5), the PCIe controller frequently drops or fails to route these MSI interrupt lines after the first few frames.
3. **Hardware Queue Wait Failure:**
   When an interrupt is dropped:
   - `libaxldev` waits until its internal timeout expires: `[libaxldev.c:140] Failed to wait for MSI 0: Connection timed out`.
   - `waitForQueueBinaryCompletion` returns `-1` or `-1301` (`ZE_RESULT_ERROR_UNKNOWN` or `ZE_RESULT_NOT_READY`).
   - The Level-Zero runtime throws an unhandled C++ `L0::Exception`, invoking `std::terminate()` (`SIGABRT`, exit code -6).
4. **Subsequent Cascading Failures:**
   Once an MSI timeout occurs, the hardware queue remains blocked with unfinished command packets, causing subsequent restart attempts to fail even faster with code `-1301`.

---

## Fix Options

### Fix 1: Full PCIe Endpoint Unbind, Bus Rescan & Driver Reload (Immediate)

Because the kernel PCIe root complex state is hung, you must remove the endpoint, rescan the PCIe bus, and reload the driver:

```bash
sudo ./tools/reset_metis_pcie.sh
```

If you prefer to perform this manually:
```bash
# 1. Kill any lingering processes holding /dev/metis* or /dev/axl*
sudo killall -9 axtrace 2>/dev/null || true

# 2. Unbind and remove Axelera PCIe device (0000:01:00.0) from the bus
echo 1 | sudo tee /sys/bus/pci/devices/0000:01:00.0/remove

# 3. Rescan PCIe root complex
echo 1 | sudo tee /sys/bus/pci/rescan
sleep 1

# 4. Reload kernel driver module
sudo rmmod axl 2>/dev/null || sudo rmmod metis 2>/dev/null || true
sudo modprobe axl single_msi=1 dma_poll=0 irq_timeout=30 dma_timeout=10 2>/dev/null || \
sudo modprobe metis single_msi=1 dma_poll=0 irq_timeout=30 dma_timeout=10

# 5. Reload AIPU firmware
~/voyager-sdk/venv/bin/axdevice --reload-firmware
```

---

### Fix 2: Test Polled DMA Mode (`dma_poll=1`) if MSI Continues Dropping

If your RK3588 board drops interrupts consistently even after link resets, enable **DMA polling mode** in the Axelera driver so the driver polls completion registers instead of relying strictly on hardware MSI interrupts:

```bash
# Temporarily enable via sysfs:
echo 1 | sudo tee /sys/module/axl/parameters/dma_poll 2>/dev/null || \
echo 1 | sudo tee /sys/module/metis/parameters/dma_poll 2>/dev/null
```
Or reload the module with `dma_poll=1`:
```bash
sudo rmmod axl 2>/dev/null || sudo rmmod metis 2>/dev/null || true
sudo modprobe axl single_msi=1 dma_poll=1 irq_timeout=30 dma_timeout=10 2>/dev/null || \
sudo modprobe metis single_msi=1 dma_poll=1 irq_timeout=30 dma_timeout=10
```

---

### Fix 3: Permanently Disable PCIe ASPM on RK3588 (Root Prevention)

Active State Power Management (ASPM) L0s/L1 transitions put the RK3588 PCIe PHY into low-power states, during which the controller misses the Metis interrupt edges.

1. **Disable ASPM in Boot Arguments:**
   Edit `/boot/armbianEnv.txt` or `/boot/extlinux/extlinux.conf`:
   ```text
   extraargs=pcie_aspm=off
   ```
   Reboot the board (`sudo reboot`).

2. **Disable ASPM via `setpci` on the Axelera Endpoint:**
   Run:
   ```bash
   sudo setpci -d 1f9d: CAP_EXP+0x10.w=0x0000
   ```
   Or use the built-in diagnostic tool:
   ```bash
   python tools/diag_metis.py --fix-aspm
   ```

3. **Verify Operating Parameters:**
   ```bash
   # Verify ASPM is disabled (should show 'performance' in brackets)
   cat /sys/module/pcie_aspm/parameters/policy
   
   # Verify single_msi is active
   cat /sys/module/axl/parameters/single_msi 2>/dev/null || cat /sys/module/metis/parameters/single_msi
   ```
