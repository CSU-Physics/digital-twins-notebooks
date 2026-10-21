# SECOM data (Module 2, part A)

These files are the SECOM data set from the UCI Machine Learning Repository:

McCann, M. and Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305

License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

| File | What it is | Change made for this course |
|---|---|---|
| `secom.data.gz` | 1,567 production records, one per line, 590 signal values separated by spaces; missing values are written `NaN` | Compressed with gzip so that it loads faster in the browser. Unzipped, it is byte for byte the file `secom.data` from UCI (SHA-256 `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c`). |
| `secom_labels.data` | one line per record: `-1` (pass) or `1` (fail) in the in-house line test, and the date and time of that test | None |
| `secom.names` | the donors' description of the data set | None |

UCI's page lists 591 attributes; the data file has 590 signal columns. The labels and their time stamps are in the second file.

The signals are anonymized: the data set does not say which tool, sensor or wafer a value comes from. The course uses it for screening signals, class imbalance and validation only.

Downloaded from https://archive.ics.uci.edu/dataset/179/secom on 20 October 2026 (`secom.zip`, SHA-256 `eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e`).
