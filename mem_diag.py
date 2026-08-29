"""
プロセスのメモリ使用量(RSS相当)をクロスプラットフォームで取得するヘルパー。

これまで metrics.py / video_io.py がそれぞれ resource.getrusage() を直接使っていたが、
resource モジュールはUnix専用でWindowsには存在しない。Windows環境ではImportErrorを
握りつぶして常に-1.0を返していたため、診断ログのRSSが常に「-1.0MB -> -1.0MB (delta+0.0MB)」
となり実質的に何も測れていなかった(この事象を修正するために切り出したモジュール)。

その後、Windows用にctypes経由でGetProcessMemoryInfoを呼ぶ実装に変更したが、
それでも-1.0のままだったため、より実績のあるpsutilを最優先で使うようにした。
psutilが無い場合のみctypesのフォールバックを使う(型指定を厳密化し、失敗時は
DEBUG_MEMORY時に理由を出力するようにして、次に問題が起きても切り分けやすくした)。
"""

from __future__ import annotations

import os
import sys

DEBUG_MEMORY = os.environ.get("COLMAP_TOOL_DEBUG_MEMORY", "0") == "1"

_warned_no_backend = False


def current_rss_mb() -> float:
    """現在のプロセスのRSS(常駐メモリ量)をMB単位で返す。取得できない場合は-1.0。"""
    global _warned_no_backend

    # 1. psutil (最も実績があり、Windows/Linux/macOS全てで確実に動く)
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception as e:
        if DEBUG_MEMORY and not _warned_no_backend:
            print(f"[mem_diag] psutil が使えないため代替手段を試します: {e}", flush=True)

    # 2. OS別のフォールバック
    if sys.platform == "win32":
        value, err = _current_rss_mb_windows()
    else:
        value, err = _current_rss_mb_unix()

    if value < 0 and DEBUG_MEMORY and not _warned_no_backend:
        _warned_no_backend = True
        print(
            f"[mem_diag] メモリ使用量を取得できませんでした: {err}\n"
            f"[mem_diag] 'pip install psutil --break-system-packages' の実行を推奨します。",
            flush=True,
        )

    return value


def _current_rss_mb_unix() -> tuple[float, str]:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrssの単位はLinuxではKB、macOS(BSD系)ではbyteなので注意
        if sys.platform == "darwin":
            return usage / (1024 * 1024), ""
        return usage / 1024, ""
    except Exception as e:
        return -1.0, str(e)


def _current_rss_mb_windows() -> tuple[float, str]:
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # windll(暗黙のargtypes/restype)は64bit環境でハンドルやBOOLの扱いを誤り、
        # 常に失敗するのに例外も出ないというケースがあり得るため、WinDLL+明示的な
        # argtypes/restype指定 + use_last_error=True で確実に原因を追えるようにする。
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)

        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []

        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()

        ctypes.set_last_error(0)
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            err_code = ctypes.get_last_error()
            return -1.0, f"GetProcessMemoryInfo失敗 (WinError {err_code})"
        return counters.WorkingSetSize / (1024 * 1024), ""
    except Exception as e:
        return -1.0, str(e)


def release_unused_memory():
    """使い終わったメモリをできるだけOSへ返却するよう促す(ベストエフォート)。

    根本原因(特定の操作で一時的に大量のメモリを確保すること)自体は解消できないが、
    その後もプロセスのメモリ使用量が高止まりし続けるのを緩和する効果を期待している。
    """
    import gc
    gc.collect()

    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
            handle = kernel32.GetCurrentProcess()
            # ワーキングセットを最小化させ、使われていないページをOSに返却させる
            kernel32.SetProcessWorkingSetSize(handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        except Exception:
            pass
    else:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
