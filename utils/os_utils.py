import sys


class OS_Utils:
    @staticmethod
    def get_xdg_equiv():
        return {
            "linux": "xdg-open",
            "win32": "start",
            "cygwin": "start",
            "msys": "start",
            "darwin": "open",
            "freebsd": "open",
            "openbsd": "open",
            "aix": "open",
            # Add mappings for specific Windows versions if necessary
        }[sys.platform]
