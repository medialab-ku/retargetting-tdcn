#!/bin/bash

cd ./datasets

wget --no-check-certificate 'https://koreaoffice-my.sharepoint.com/:u:/g/personal/sang-bin_korea_edu/Ef4pz2xtsCpCrYDEd4fzT2oB62SVK3CpQyh-uBPH6ka3tg?download=1' -O test.zip
unzip test.zip
rm test.zip

cd ../

