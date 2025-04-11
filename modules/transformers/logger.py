#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import inspect
import os
import builtins as __builtins__  # Use a different name to avoid conflict


# --- Custom Print Function ---
class LoggingUtils:
    def __init__(self):
        # Store the original print function
        self.original_print = __builtins__.print

    def custom_print(self, *args, **kwargs):
        """Custom print function that includes filename and line number."""
        # Get the frame of the caller's caller (to get the location outside this logger module)
        frame = inspect.currentframe().f_back
        # Go one more step back in the stack if the call is from within this module (e.g., print = logger.custom_print line)
        # This might need adjustment depending on how print is used globally
        if frame.f_code.co_filename == __file__:
             frame = frame.f_back

        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno

        # Format the prefix
        prefix = f"[{filename}:{lineno}]"

        # Call the original print function with the prefix
        self.original_print(prefix, *args, **kwargs)

# Instantiate the logger
logger = LoggingUtils()

# Overload the built-in print globally (use with caution)
# This approach means any module importing this will potentially
# affect the global print function. A better approach might be
# to pass the logger instance explicitly where needed.
# For now, sticking to the original pattern:
print = logger.custom_print
# --- End Custom Print Function --- 