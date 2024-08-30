#!/bin/bash

# originally generated with ChatGPT-4o on 2024-08-29,
# with modifications

# Set the directory containing the graph
graph_dir="gbad/schema/authority_AgentControlRelation"

# Find the first *.drawio file in the graph directory
drawio_file=$(find "$graph_dir" -type f -name "*.drawio" | head -n 1)

# Define the range of files to generate
start=1  # Starting suffix
end=14   # Ending suffix

# Loop through the desired range
for i in $(seq $((start+1)) $end); do
  # Read the content of the original .drawio file
  file_content=$(<"$drawio_file")

  # Make the necessary replacements, using the current value of i
  updated_content=$(echo "$file_content" | \
    sed "s/RiC-O_1-0-1\/{HEADING}\/AgentControlRelation\/${start}/RiC-O_1-0-1\/{HEADING}\/AgentControlRelation\/${i}/g" | \
    sed "s/RiC-O_1-0-1\/{CONTAG_${start}}\/CorporateBody\/1/RiC-O_1-0-1\/{CONTAG_${i}}\/CorporateBody\/1/g" | \
    sed "s/KB\/Date\/{DATECONT_${start}_END}/KB\/Date\/{DATECONT_${i}_END}/g" | \
    sed "s/KB\/Date\/{DATECONT_${start}_BEGINNING}/KB\/Date\/{DATECONT_${i}_BEGINNING}/g")

  # Define the output file name with the new suffix
  output_file="${drawio_file%.drawio}_${i}.drawio"

  # Save the modified content to the new file
  echo "$updated_content" > "$output_file"

  echo "Generated $output_file with replacements."
done
