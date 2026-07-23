# P210 day-one runbook

What to do the day the Neptune arrives, in an order that cannot brick the board
and that produces real value immediately. This is the execution plan behind the
gap register; it starts with the safe, bitstream-free path and gates every
irreversible step behind a check.

## Gate zero (do these before anything irreversible)

Confirmed from the committed `system_top.xsa` (`gate-zero` inspection):

- **The factory PL is a standard ADI HDL design and is editable in principle.**
  The XSA carries `system.hwh` (full hardware handoff), `system.bda`, a bitstream
  (`system_top.bit`), and the IP inventory: `analog.com:user:axi_ad9361`,
  `axi_dmac` (adc + dac), `axi_tdd`, `util_cpack2/upack2`, a `fir_compiler`, and
  the PS7 at processing_system7 5.5. The operator accelerator would be added to a
  Vivado block design rebuilt from this, sharing the PS7 HP DDR port with the two
  existing `axi_dmac` engines. The accelerator register window (0x7C450000) sits
  cleanly above the occupied AXI-GP range (0x7C400000..0x7C440000 are used).
  **Still to confirm on the physical board:** that the exact IP versions are
  reproducible in your Vivado install (ADI HDL tag matching the P210 release) and
  that a rebuilt bitstream boots the radio unchanged.
- **Get a revision-matched recovery image and preserve the factory SD media
  before loading any custom bitstream.** The firmware posture is `flashable:
  false` for a reason.
- **Confirm a volatile JTAG load path.** Load a test bitstream into PL config
  over JTAG (no flash write) and confirm the radio still enumerates. If the P210
  boots secure/locked such that volatile PL load is blocked, in-fabric bring-up
  is gated until that is resolved -- do not proceed to flashing.

## Step 1 -- host-side operator on real captures (safe, immediate)

No bitstream, no flashing, nothing that can brick the board.

1. Bring the board up on its factory firmware; confirm RX via standard `iiod`
   (TCP 30431). Pin the AGC to **manual gain (MGC)** for any characterization --
   every impairment estimate is per-gain-index and a moving AGC invalidates them.
2. Capture wideband IQ (`capture.BytesCaptureSource` decodes the board's signed
   IQ16 interleaved format directly).
3. Run a pretrained operator bank host-side (`field.run_on_capture`, or the
   bit-exact `golden_executor.GoldenExecutor`). This is the operator doing real
   work on real signals on day one.

## Step 2 -- characterize the radio (feeds pretraining)

Transmit the known calibration waveform (`impairments.calibration_waveform`)
over a **cabled, attenuated** TX->RX loopback (or the AD9361 internal BIST
loopback -- never an open RF path into the RX front end), capture it, and run
`impairments.estimate_impairments`. This returns DC offset, IQ gain/phase
imbalance, CFO, and noise floor -- the AD9361 effects the gym does not yet model.
Fold the measured values into the gym so pretraining matches your actual radio.
The estimators are verified in sim against planted values (`test_impairments`).

## Step 3 -- known-signal fine-tune (reliable adaptation)

Fine-tune the operator to your actual channel using the **known reference**
(`field.finetune_known_signal`): loopback captures carry truth, so the coherence
reward applies and adaptation is reliable, with a validation gate that refuses to
ship a bank that does not strictly improve.

> Do **not** attempt blind, over-the-air fine-tuning. It was measured to degrade
> true signal quality on most channels (the blind proxy is only moderately
> correlated with truth). Reliable adaptation requires a known reference.

## Step 4 -- in-fabric (only after gate zero clears)

Freeze golden-arithmetic v1 (done: `specs/golden-arithmetic-v1.md`), build the
operator accelerator into a Vivado block design rebuilt from the XSA, verify it
in RTL simulation against the golden vectors (the FFT engine already passes),
close timing/resources in Vivado on the xc7z020 alongside the radio (the open
hardware gap), and JTAG-load it volatile first. The QEMU twin and the C core
already reproduce the golden arithmetic bit-for-bit, so the bitstream has a
proven reference to match before it is ever trusted.

## What is proven vs open

Proven in software today: the golden integer arithmetic (Python == C == RTL on
shared vectors); the coherence loop closing through the fully-integer datapath;
the capture byte format; the impairment estimators; the known-signal fine-tune
with its validation gate.

Open until hardware/Vivado: bitstream resource and timing closure on the
xc7z020; the QEMU MMIO device wrapper around the proven C core; the actual radio
behaviour under a rebuilt bitstream; and the AD9361 impairment magnitudes (which
the Step 2 harness measures on arrival). See the gap register for the full list.
