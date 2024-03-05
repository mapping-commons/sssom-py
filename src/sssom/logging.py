"""Logging utilities for sssom-py."""

import logging
from logging import NOTSET, WARNING, Logger


class SSSOMLogger(Logger):
    """Custom logger for SSSOM."""

    def __init__(self, name, level=NOTSET, truncate_log_messages=20):
        """Construct SSSOM logger, including truncate_log_messages parameter to truncate messages."""
        super().__init__(name, level)
        self.count_logs = {}
        self.truncate_log_messages = truncate_log_messages

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        # You can check the current log level here and modify the message or add extra information.

        log_id = extra.get("log_id") if extra else None
        if log_id:
            self._count_up(log_id)
            log_type_printed_count = self.count_logs[log_id]
        else:
            log_type_printed_count = 0

        block_logging = False

        if level >= self.level:
            # Add custom behavior based on the log level
            if level <= WARNING and log_type_printed_count > self.truncate_log_messages:
                # If the loglevel is WARNING or less (DEBUG, INFO) and we have already recorded more than
                # the configured truncate_log_messages limit
                block_logging = True

        # Call the original _log method to actually log the message
        if not block_logging:
            super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

    def _count_up(self, log_id):
        if log_id in self.count_logs:
            self.count_logs[log_id] = self.count_logs[log_id] + 1
        else:
            self.count_logs[log_id] = 1

    def print_summary(self):
        """Log a warning if any errors were truncated."""
        outstring = "Summary of errors logged.+\n"
        truncated = False
        for log_id in self.count_logs:
            truncated = True
            outstring += f"{log_id}: {self.count_logs[log_id]} errors\n"
        if self.level <= WARNING and truncated:
            outstring += f"Some errors which were logged more than {self.truncate_log_messages} times were truncated."
        if truncated:
            logging.warning(outstring)
