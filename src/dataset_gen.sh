# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#!/usr/bin/env bash
trdg -c 3356 -f 64 -sym -l en -t 8 -na 1 -rbl -rk --output_dir ./data/dataset
python src/DatasetGenerator.py --dataset_path ./data
