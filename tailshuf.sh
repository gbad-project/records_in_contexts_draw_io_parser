#!/bin/bash

# Originally generated with ChatGPT 4o on 2024-11-16, modified

# Define the prespecified directory
source_dir="gbad/mapping/source"

# Ensure correct number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <filename> -n <number_of_lines>"
    exit 1
fi

# Input arguments
filename="$1"
input_file="${source_dir}/${filename}"
shift  # Shift to access the '-n' argument and the number
while getopts ":n:" opt; do
    case $opt in
        n) num_lines=$OPTARG ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Ensure the number of lines is provided
if [ -z "$num_lines" ]; then
    echo "Error: You must specify the number of lines with -n."
    exit 1
fi

# Extract filename components
extension="${filename##*.}"                          # Get the extension
name="${filename%.*}"                                # Get the name without extension
lowercase_name=$(echo "$name" | tr '[:upper:]' '[:lower:]')  # Convert name to lowercase

# Construct the output filename
output_file="${source_dir}/${lowercase_name}_tailshuf_${num_lines}.${extension}"

# Capture the first line of the input file
header=$(head -n 1 "$input_file")

# Perform random sampling, prepend the header, and write to the output file
{
    echo "$header"    # Prepend the first line
    tail -n +2 "$input_file" | shuf -n "$num_lines"  # Skip first line and sample
} > "$output_file"

echo "Random sample of $num_lines lines saved to $output_file"
