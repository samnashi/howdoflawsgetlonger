echo "convlstm for Batch Size = 1024 and Generator Pad Size = 128"
echo "running the high learning rate, 300 draw sequence"
echo "HLR done"
echo "running the medium learning rate, 300 draw sequence"
python convlstm_1024_128_MLR.py
echo "MLR done"
echo "running the low learning rate, 600 draw sequence"
python convlstm_1024_128_LLR.py
echo "LLR done"
echo "fin."
