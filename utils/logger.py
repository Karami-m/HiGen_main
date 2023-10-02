import logging
import sys


def setup_logging(log_level, log_file, logger_name="exp_logger"):
  """ Setup logging """
  numeric_level = getattr(logging, log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % log_level)

  if sys.version_info[0] >=3 and sys.version_info[1] >=8: # for python version >= 3.8
      logging.basicConfig(
          filename=log_file,
          filemode="w",
          format="%(levelname)-5s | %(asctime)s | File %(filename)-20s | Line %(lineno)-5d | %(message)s",
          datefmt="%m/%d/%Y %I:%M:%S %p",
          level=numeric_level,
          force = True
      )
  else:
      logging.basicConfig(
          filename=log_file,
          filemode="w",
          format="%(levelname)-5s | %(asctime)s | File %(filename)-20s | Line %(lineno)-5d | %(message)s",
          datefmt="%m/%d/%Y %I:%M:%S %p",
          level=numeric_level,
      )

  # to define a Handler which writes messages to the sys.stderr
  console = logging.StreamHandler()
  console.setLevel(numeric_level)
  # set a format which is simpler for console use
  formatter = logging.Formatter(
      "%(levelname)-5s | %(asctime)s | %(filename)-25s | line %(lineno)-5d: %(message)s"
  )
  # tell the handler to use this format
  console.setFormatter(formatter)
  # add the handler to the root logger
  logging.getLogger(logger_name).addHandler(console)

  return get_logger(logger_name)


def get_logger(logger_name="exp_logger"):
  return logging.getLogger(logger_name)
