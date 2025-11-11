for app_dir in mobile_eval_e_retrieve/operator/app/*; do
    # Check if the directory exists and contains passage.tsv
    tsv_file="$app_dir/passage.tsv"
    embedding_dir="$app_dir/embedding"
    embedding_file="$embedding_dir/passages_00"

    if [ -f "$tsv_file" ]; then
        echo "Checking $app_dir..."

        # Check if embedding file exists and compare modification times
        if [ ! -f "$embedding_file" ] || [ "$tsv_file" -nt "$embedding_file" ]; then
            echo "Updating embeddings for $app_dir: passage.tsv is new or embedding file is missing."
            python generate_passage_embeddings.py \
                --model_name_or_path contriever-msmarco \
                --passages "$tsv_file" \
                --output_dir "$embedding_dir"
        else
            echo "Skipped $app_dir: passage.tsv has not changed since last embedding."
        fi
    else
        echo "Skipped $app_dir: passage.tsv not found."
    fi
done

