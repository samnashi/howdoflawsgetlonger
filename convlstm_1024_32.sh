#!/usr/bin/env bash
echo "convlstm for Batch Size = 1024 and Generator Pad Size = 32"
echo "running the high learning rate, 300 draw sequence"
python convlstm_1024_32_HLR.py
echo "HLR done"
echo "running the medium learning rate, 300 draw sequence"
python convlstm_1024_32_MLR.py
echo "MLR done"
echo "running the low learning rate, 600 draw sequence"
python convlstm_1024_32_LLR.py
echo "LLR done"
echo "fin."
