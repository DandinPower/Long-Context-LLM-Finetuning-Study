#!/bin/bash

# Initialize variables for tracking memory usage
initial_mem=0
max_mem=0
max_mem_timestamp=""
initial_timestamp=$(date "+%Y-%m-%d %H:%M:%S.%N")

# Function to convert KB to MB and GB
format_memory() {
    local mem_kb=$1
    local mem_mb=$(echo "scale=2; $mem_kb/1024" | bc)
    local mem_gb=$(echo "scale=2; $mem_kb/1024/1024" | bc)
    echo "${mem_mb}MB (${mem_gb}GB)"
}

# Get initial memory usage
initial_mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
initial_used=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') - initial_mem))
echo "Initial memory usage: $(format_memory $initial_used)"

while true; do
    # Get current timestamp
    current_timestamp=$(date "+%Y-%m-%d %H:%M:%S.%N")

    # Get current memory usage
    mem_available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    current_used=$((mem_total - mem_available))

    # Update maximum memory usage and its timestamp
    if [ $current_used -gt $max_mem ]; then
        max_mem=$current_used
        max_mem_timestamp=$current_timestamp
    fi

    current_stage_used=$((current_used - initial_used))
    maximum_stage_used=$((max_mem - initial_used))

    # Clear screen and display results
    clear
    echo "Time Statistics:"
    echo "Initial Time: ${initial_timestamp:0:23}"
    echo "Max Memory Time: ${max_mem_timestamp:0:23}"
    echo "Current Time: ${current_timestamp:0:23}"
    echo ""
    echo "Memory Statistics:"
    echo "Initial memory usage: $(format_memory $initial_used)"
    echo "Maximum memory usage: $(format_memory $max_mem)"
    echo -e "\e[31mMaximum Monitor memory usage: $(format_memory $maximum_stage_used)\e[0m"
    echo "Current memory usage: $(format_memory $current_used)"
    echo "Current Monitor memory usage: $(format_memory $current_stage_used)"
    
    # Sleep for a very short duration (100 microseconds)
    sleep 0.0001
done