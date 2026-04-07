#!/usr/bin/env python3
"""Simple trigger I/O tester for Hikvision industrial cameras.

This script keeps the camera in continuous acquisition while configuring one
hardware line as a digital input (for detecting the fire signal) and another as
an opto-isolated output (for driving the water valve relay).  It can be used to
manually pulse the output and watch the input status to verify wiring.
"""

import argparse
import sys
import threading
import time
from ctypes import POINTER, c_bool, c_ubyte, cast


sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from CameraParams_const import MV_USB_DEVICE  # noqa: E402
from CameraParams_header import (  # noqa: E402
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_FRAME_OUT_INFO_EX,
)
from MvCameraControl_class import MvCamera  # noqa: E402
from MvErrorDefine_const import MV_OK  # noqa: E402


class HikIoTester:
    """Wraps the bare SDK calls that we need for IO verification."""

    def __init__(self):
        self.cam = MvCamera()
        self.device_info = None
        self.buffer_size = 40 * 1024 * 1024
        self.buf = None
        self._grab_thread = None
        self._grabbing = False
        self.frame_info = MV_FRAME_OUT_INFO_EX()
        self.frame_count = 0
        self.input_line = "Line0"
        self.output_line = "Line2"

    # ------------------------------------------------------------------
    # Camera lifecycle helpers
    # ------------------------------------------------------------------
    def open(self):
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, device_list)
        if ret != MV_OK or device_list.nDeviceNum == 0:
            print("[ERR] No camera found, ret=0x%x" % ret)
            return False

        self.device_info = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(self.device_info)
        if ret != MV_OK:
            print("[ERR] CreateHandle failed: 0x%x" % ret)
            return False

        ret = self.cam.MV_CC_OpenDevice(0, 0)
        if ret != MV_OK:
            print("[ERR] OpenDevice failed: 0x%x" % ret)
            self.cam.MV_CC_DestroyHandle()
            return False

        # Keep the camera in free-run mode so frames are always available.
        self.cam.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous")
        self.cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
        return True

    def start_stream(self):
        if self._grabbing:
            return

        self.buf = (c_ubyte * self.buffer_size)()
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            raise RuntimeError("StartGrabbing failed: 0x%x" % ret)

        self._grabbing = True
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def _grab_loop(self):
        info = MV_FRAME_OUT_INFO_EX()
        while self._grabbing:
            ret = self.cam.MV_CC_GetOneFrameTimeout(self.buf, self.buffer_size, info, 500)
            if ret == MV_OK:
                self.frame_info = info
                self.frame_count += 1
            else:
                time.sleep(0.01)

    def close(self):
        self._grabbing = False
        if self._grab_thread:
            self._grab_thread.join(timeout=1)
        if self.cam:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self.cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # IO configuration and helpers
    # ------------------------------------------------------------------
    def configure_lines(self, input_line="Line0", output_line="Line2", debounce_us=2000):
        self.input_line = input_line
        self.output_line = output_line

        # Configure input line: opto input connected to the fire detection signal.
        self.cam.MV_CC_SetEnumValueByString("LineSelector", input_line)
        self.cam.MV_CC_SetEnumValueByString("LineMode", "Input")
        self.cam.MV_CC_SetBoolValue("LineInverter", False)
        self.cam.MV_CC_SetFloatValue("LineDebouncerTime", float(debounce_us))

        # Keep trigger disabled so acquisition is continuous, but we can still
        # watch the line status and optionally map it as a trigger in the future.
        self.cam.MV_CC_SetEnumValueByString("TriggerActivation", "RisingEdge")
        self.cam.MV_CC_SetEnumValueByString("TriggerSource", input_line)

        # Configure output line to be driven by UserOutput0, which we toggle
        # through UserOutputValue in the SDK.
        self.cam.MV_CC_SetEnumValueByString("UserOutputSelector", "UserOutput0")
        self.cam.MV_CC_SetBoolValue("UserOutputValue", False)
        self.cam.MV_CC_SetEnumValueByString("LineSelector", output_line)
        self.cam.MV_CC_SetEnumValueByString("LineMode", "Output")
        self.cam.MV_CC_SetEnumValueByString("LineSource", "UserOutput0")

    def read_input_level(self):
        self.cam.MV_CC_SetEnumValueByString("LineSelector", self.input_line)
        status = c_bool(False)
        ret = self.cam.MV_CC_GetBoolValue("LineStatus", status)
        if ret != MV_OK:
            raise RuntimeError("LineStatus read failed: 0x%x" % ret)
        return status.value

    def pulse_output(self, active_ms=500):
        self.cam.MV_CC_SetEnumValueByString("UserOutputSelector", "UserOutput0")
        self.cam.MV_CC_SetBoolValue("UserOutputValue", True)
        time.sleep(active_ms / 1000.0)
        self.cam.MV_CC_SetBoolValue("UserOutputValue", False)

    # ------------------------------------------------------------------
    # Composite tests
    # ------------------------------------------------------------------
    def run_test(self, iterations=3, pulse_ms=500, idle_interval=2.0):
        for idx in range(iterations):
            print(f"[TEST] Iteration {idx + 1}/{iterations}")
            before = self.read_input_level()
            print(f"       Input before pulse: {'HIGH' if before else 'LOW'}")
            self.pulse_output(pulse_ms)
            time.sleep(0.05)
            after = self.read_input_level()
            print(f"       Input after pulse: {'HIGH' if after else 'LOW'}")
            time.sleep(idle_interval)
        print(f"Captured {self.frame_count} frames while testing.")


def parse_args():
    parser = argparse.ArgumentParser(description="Test Hikvision camera IO lines")
    parser.add_argument("--input-line", default="Line0", help="Line used for fire detection input (default Line0)")
    parser.add_argument("--output-line", default="Line2", help="Line used to drive the water valve relay (default Line2)")
    parser.add_argument("--debounce-us", type=float, default=2000.0, help="Debounce time for input line in microseconds")
    parser.add_argument("--pulse-ms", type=int, default=1000, help="How long the output stays HIGH per test pulse")
    parser.add_argument("--idle", type=float, default=2.0, help="Idle time between output pulses (seconds)")
    parser.add_argument("--iterations", type=int, default=5, help="How many pulses to send before stopping")
    return parser.parse_args()


def main():
    args = parse_args()
    tester = HikIoTester()
    if not tester.open():
        sys.exit(1)

    tester.configure_lines(args.input_line, args.output_line, args.debounce_us)
    tester.start_stream()

    try:
        tester.run_test(args.iterations, args.pulse_ms, args.idle)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        tester.close()


if __name__ == "__main__":
    main()
