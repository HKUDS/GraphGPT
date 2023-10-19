# to fill in the following path to extract projector for the second tuning stage!
src_model=
output_proj=

python3.8 ./scripts/extract_graph_projector.py \
  --model_name_or_path ${src_model} \
  --output ${output_proj}