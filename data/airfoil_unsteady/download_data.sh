# Download the airfoil_unsteady dataset (MeshGraphNets).
# Run as: download_data.sh airfoil <path>
# Example: ./download_data.sh airfoil /path/to/data  -> downloads to /path/to/data/airfoil/
set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/${DATASET_NAME}"

BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}/"

mkdir -p ${OUTPUT_DIR}
for file in meta.json train.tfrecord valid.tfrecord test.tfrecord
do
wget -O "${OUTPUT_DIR}/${file}" "${BASE_URL}${file}"
done
