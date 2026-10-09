# Research data and code

This repository contains four data files and one Python analysis script deposited by [chunai3270](https://github.com/chunai3270).

## Files

| File | Format | Size (bytes) |
| --- | --- | ---: |
| [Electronic Nose.xlsx](Electronic%20Nose.xlsx) | Excel (.xlsx) | 13620 |
| [Machine Learning.CSV](Machine%20Learning.CSV) | CSV | 3920 |
| [Machine Learning.py](Machine%20Learning.py) | Python script | 14815 |
| [Raw Data.xls](Raw%20Data.xls) | Excel (.xls) | 37376 |
| [Single sensory data example.xlsx](Single%20sensory%20data%20example.xlsx) | Excel (.xlsx) | 9541 |

The original file names and file contents are preserved.

## Analysis script

`Machine Learning.py` contains an SVR analysis using scikit-learn, Optuna parameter optimization and SHAP explanations.

The script imports these third-party packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `optuna`, and `shap`.

Before running, review the configuration block at the beginning of the script:

- `DATA_FILE` currently points to `chili_oil_features.csv`, which is not included under that name in this repository. Set it to the intended input file after confirming the appropriate dataset.
- `TARGET_COLUMN` defaults to `grade`. Confirm the target column name, label encoding, standardization setting and any historical SVR parameters against the original analysis.
- The input validation expects nine rows and three numeric labels with three rows per label.

Run the configured script from the directory containing your input data:

```bash
python "Machine Learning.py"
```

The configured output directory is `SVR_SHAP_results`. This upload was checked for file integrity; execution and reproduction of manuscript results have not been validated.

## Download

Download individual files using the links above, or choose **Code → Download ZIP** to download the repository.

## Integrity

`CHECKSUMS.sha256` records the SHA-256 checksum of each original data file and the Python script.

## Availability and version

Repository: https://github.com/chunai3270/data

For a manuscript, link to the specific commit containing the data and code used in the study, so readers can retrieve the same version.

## 中文说明

本仓库存放 4 个数据文件和 1 个 Python 分析脚本，保留上传时的原文件名与文件内容。可点击上表下载单个文件，或使用 **Code → Download ZIP** 下载整个仓库。`CHECKSUMS.sha256` 提供各原始文件的 SHA-256 校验值。

`Machine Learning.py` 使用 SVR、Optuna 和 SHAP 进行分析。运行前需检查脚本开头的配置：默认输入文件为 `chili_oil_features.csv`，仓库中没有这个名称的文件；请先确认对应数据，再设置 `DATA_FILE`、目标列和分析参数。输入检查要求 9 行数据、3 个数值标签且每组 3 行。本次已验证上传文件完整性，尚未验证脚本运行及论文结果复现。

论文引用时可使用对应提交的固定版本链接。
