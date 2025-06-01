#!/bin/bash

# --- Configuration ---
CELL_SIZE=250 # Ouput size for each square video (e.g., 250 means 250x250)
OUTPUT_FILENAME="output_grid.mp4"
# --- End Configuration ---

# Find MP4 files, sort them
shopt -s nullglob
files_unsorted=(*.mp4)
# Sort files for consistent ordering (optional, but good practice)
IFS=$'\n' files=($(sort <<<"${files_unsorted[*]}"))
unset IFS

# Check if we have exactly 10 files
if [ "${#files[@]}" -ne 10 ]; then
  echo "Error: Need exactly 10 MP4 files for a 5x2 grid. Found ${#files[@]}."
  if [ "${#files[@]}" -gt 0 ]; then
    echo "Files found:"
    for f in "${files[@]}"; do echo "  - $f"; done
  fi
  exit 1
fi

echo "Processing the following 10 files in this order for the grid:"
for i in $(seq 0 9); do
  echo "  $((i+1)). ${files[$i]}"
done
echo "Each cell will be ${CELL_SIZE}x${CELL_SIZE}."

ffmpeg_inputs=""
video_scale_chains=""
video_labels_for_xstack=""
audio_input_labels_for_amix=""
audio_pre_processing="" # For generating silent audio if needed

for i in $(seq 0 9); do
  current_file="${files[$i]}"
  ffmpeg_inputs+="-i \"${current_file}\" "

  # Video processing
  video_scale_chains+="[${i}:v]scale=${CELL_SIZE}:${CELL_SIZE},setpts=PTS-STARTPTS[v${i}]; "
  video_labels_for_xstack+="[v${i}]"

  # Audio processing: Check if audio stream exists
  if ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 "${current_file}" | grep -q "audio"; then
    echo "File '${current_file}' has audio. Using [${i}:a]."
    audio_input_labels_for_amix+="[${i}:a]"
  else
    echo "File '${current_file}' has NO audio. Generating silent track [silent${i}]."
    # Create a label for a silent audio stream for this input
    # This silent stream will be generated once and used by amix
    # The duration of anullsrc will be handled by amix's 'duration' option.
    audio_pre_processing+="anullsrc=channel_layout=stereo:sample_rate=44100[silent${i}]; "
    audio_input_labels_for_amix+="[silent${i}]"
  fi
done

# The xstack layout for 5x2 (w0, w1 etc. will use CELL_SIZE implicitly)
XSTACK_LAYOUT="0_0|w0_0|w0+w1_0|w0+w1+w2_0|w0+w1+w2+w3_0|0_h0|w0_h0|w0+w1_h0|w0+w1+w2_h0|w0+w1+w2+w3_h0"

# Construct the full ffmpeg command
# Note: audio_pre_processing is placed at the beginning of the filter_complex string
ffmpeg_command="ffmpeg ${ffmpeg_inputs} \
-filter_complex \"\
${audio_pre_processing}\
${video_scale_chains}\
${video_labels_for_xstack}xstack=inputs=10:layout=${XSTACK_LAYOUT}[vout]; \
${audio_input_labels_for_amix}amix=inputs=10:duration=first:dropout_transition=3[aout]\
\" \
-map \"[vout]\" -map \"[aout]\" -c:v libx264 -crf 23 -preset veryfast \"${OUTPUT_FILENAME}\""

echo ""
echo "Executing FFmpeg command:"
echo "${ffmpeg_command}"
echo ""

# Execute the command
eval "${ffmpeg_command}"

if [ $? -eq 0 ]; then
  echo ""
  echo "FFmpeg processing complete. Output should be ${OUTPUT_FILENAME}"
else
  echo ""
  echo "FFmpeg processing failed."
fi