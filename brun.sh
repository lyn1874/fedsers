#!/bin/bash
#SBATCH --job-name=sers
#SBATCH --output=sers-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mail-user=blia@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL


for v in 3
do
    lr_init=0.1
    aggregation=fed_avg 
    model_arch=vit #VGG_12

    partition_type=non_iid 
    n_clients=8
    loc_n_epoch=10
    end_commu=60

    # partition_type=centralised
    # loc_n_epoch=200 
    # n_clients=1
    # end_commu=0

    start=0
    dataset=sers 
    num_class=2
    loc=nobackup
    batch_size=64

    lr_schedule=constant
    num2=4
    repeat_gpu=4

    base=0 


    visible_gpu="$CUDA_VISIBLE_DEVICES"
    IFS=',' read -ra gpu_array <<< "$visible_gpu"
    echo "GPU array: ${gpu_array[@]}"

    for j in $(seq "$start" 1 "$end_commu")
    do
        for i in $(seq 0 1 "$((n_clients-1))")
        do
            if [ "$i" -lt "$num2" ]; then
                gpu_index="$((i+base))"
            elif [ "$i" -ge "$num2" ]; then 
                gpu_index="$((i-repeat_gpu+base))"
            fi
            echo "GPU index: ${gpu_array[$gpu_index]}"
            export CUDA_VISIBLE_DEVICES="${gpu_array[$gpu_index]}"
            # export CUDA_VISIBLE_DEVICES="0"
            
            python3 create_train.py --use_local_id "$i" --local_n_epochs "$loc_n_epoch" \
                --lr "$lr_init" \
                --communication_round "$j" --loc "$loc" \
                --lr_schedule "$lr_schedule" --version "$v" --arch "$model_arch" \
                --partition_type "$partition_type" \
                --data "$dataset" --num_class "$num_class" \
                --batch_size "$batch_size" \
                --n_clients "$n_clients" --aggregation "$aggregation" &
        done
        wait 
        echo "Done training all the clients"
        for i in $(seq 0 1 "$((num2-1))")
        do
            export CUDA_VISIBLE_DEVICES="${gpu_array[$gpu_index]}"
            # export CUDA_VISIBLE_DEVICES="0"

            if [ "$i" == 0 ]; then 
                worker_for_occupy_gpu=false
            else
                worker_for_occupy_gpu=true 
            fi 
            python3 communicate.py --use_local_id "$i" --local_n_epochs "$loc_n_epoch" \
                --lr "$lr_init" \
                --communication_round "$j" --loc "$loc" \
                --lr_schedule "$lr_schedule" --version "$v" --arch "$model_arch" \
                --partition_type "$partition_type" --num_class "$num_class" \
                --data "$dataset" --worker_for_occupy_gpu "$worker_for_occupy_gpu" \
                --n_clients "$n_clients" --aggregation "$aggregation" &
        done
        wait 
        echo "Done communicating"
    done
done 