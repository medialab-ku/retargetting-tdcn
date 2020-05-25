#!/bin/bash

wget --no-check-certificate 'https://koreaoffice-my.sharepoint.com/:u:/g/personal/sang-bin_korea_edu/ETgOb6Wkp2hBvQe_tGzljggBVl9IO2tmqbVjGKWwMlS6cg?download=1' -O best.ckpt

mkdir -p ./ckpt/ours/retargetting-tdcn/gen/param
mv best.ckpt ./ckpt/ours/retargetting-tdcn/gen/param

python evaluate.py --config=ours.retargetting-tdcn