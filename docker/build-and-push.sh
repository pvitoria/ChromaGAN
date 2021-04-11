#!/bin/bash -eux

# Run this script from the repo's root folder
#
# $ ./docker/build-and-push.sh

# 1. Build Docker images for CPU and GPU

image="us-docker.pkg.dev/replicate/pvitoria/chromagan"
cpu_tag="$image:cpu"
gpu_tag="$image:gpu"
docker build -f docker/Dockerfile.cpu --tag "$cpu_tag" .
docker build -f docker/Dockerfile.gpu --tag "$gpu_tag" .

# 2. Test Docker images

test_input_folder=/tmp/test-chromagan/input
mkdir -p $test_input_folder
cp SAMPLE_IMGS/grayscale/bruschetta.png $test_input_folder/
test_output_folder=/tmp/test-chromagan/output

docker run -it \
        -v $test_input_folder:/code/DATASET/imagenet/test \
        -v $test_output_folder/cpu:/code/RESULT/imagenet \
        $cpu_tag

[ -f $test_output_folder/cpu/*_bruschettapsnr_reconstructed.jpg ] || exit 1

docker run -it \
        -v $test_input_folder:/code/DATASET/imagenet/test \
        -v $test_output_folder/gpu:/code/RESULT/imagenet \
        $gpu_tag

[ -f $test_output_folder/gpu/*_bruschettapsnr_reconstructed.jpg ] || exit 1

sudo rm -rf "$test_input_folder"
sudo rm -rf "$test_output_folder"

# 3. Push the Docker images

docker push $cpu_tag
docker push $gpu_tag
