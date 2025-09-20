
conda activate lhat

clear


FOLDER="data"
FILES=($(ls -1 "$FOLDER"))

echo "Available files:"
for i in "${!FILES[@]}"; do
    echo "[$((i+1))] ${FILES[$i]}"
done

read -p "Enter file number: " IDX

read -p "Use features in logarithm? (y/n): " yn

case $yn in
    [Yy]* ) LOG_X="--log_X";;
    [Nn]* ) LOG_X="";;
    * ) echo "Please answer y or n."; exit 1;;
esac

python -m rain_modeling.rain_landslide_modeling \
  --file_path $IDX \
  --x_feat intensity \
  --x_feat cumulative \
  $LOG_X