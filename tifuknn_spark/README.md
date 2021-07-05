Spark Structured Streaming application.


# Notes

- When testing File source for spark structured streaming, make sure you
use `cp` or `mv` command to add file to the target directory, otherwise
  spark might not detect the files added.