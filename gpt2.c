#include "gpt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int load_matrix(const char *filename, Matrix *matrix)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        perror("Failed to open file");
        return -1; // Return -1 on error
    }

    fread(&matrix->rows, sizeof(int), 1, f);
    fread(&matrix->cols, sizeof(int), 1, f);
    matrix->weights = malloc(matrix->rows * matrix->cols * sizeof(float));
    if (!matrix->weights)
    {
        perror("Failed to allocate memory for weights");
        fclose(f);
        return -1; // Return -1 on error
    }
    fread(matrix->weights, sizeof(float), matrix->rows * matrix->cols, f);
    fclose(f);

    // printf("\t load %s, matrix dims (%d x %d)\n", filename, matrix->rows, matrix->cols);
    return 0; // Return 0 on success
}

int get_embeddings(Matrix *emd, int *pos, int seq_len, float *hidden_state)
{
    // Initialize hidden state to zero
    memset(hidden_state, 0, seq_len * emd->cols * sizeof(float));

    // Get token embeddings
    for (int i = 0; i < seq_len; i++)
    {
        memcpy(hidden_state + i * emd->cols,
               emd->weights + pos[i] * emd->cols,
               emd->cols * sizeof(float));
    }

    return 0; // Return 0 on success
}

void mat_layer_norm(const Matrix *input, Matrix *output, float bias, float eps)
{
    for (int i = 0; i < input->rows; i++)
    {
        float mean = 0.0f;
        float variance = 0.0f;

        // Calculate mean
        for (int j = 0; j < input->cols; j++)
        {
            mean += input->weights[i * input->cols + j];
        }
        mean /= input->cols;

        // Calculate variance
        for (int j = 0; j < input->cols; j++)
        {
            float diff = input->weights[i * input->cols + j] - mean;
            variance += diff * diff;
        }
        variance /= input->cols;

        // Normalize and write to output
        for (int j = 0; j < output->cols; j++)
        {
            output->weights[i * output->cols + j] = ((input->weights[i * input->cols + j] - mean) / sqrt(variance + eps)) + bias;
        }
    }
}

void attention(Matrix *Q, Matrix *K, Matrix *V, Matrix *mask, Matrix *output)
{
    // This function should implement the attention mechanism
    // For now, we will just print a message
    //printf("Processing attention with Q: %dx%d, K: %dx%d, V: %dx%d\n",
    //       Q->rows, Q->cols, K->rows, K->cols, V->rows, V->cols);

    int dim_k = K->cols;

    //printf("Dim K %d\n", dim_k);

    Matrix scores;
    scores.rows = Q->rows;
    scores.cols = K->rows;
    scores.weights = malloc_and_check(scores.rows * scores.cols * sizeof(float));
    // Calculate attention scores

    Matrix sof_scores;
    sof_scores.rows = Q->rows;
    sof_scores.cols = K->rows;
    sof_scores.weights = malloc_and_check(sof_scores.rows * sof_scores.cols * sizeof(float));

    Matrix K_T;
    K_T.weights = malloc_and_check(K->rows * K->cols * sizeof(float));
    K_T.rows = K->cols;
    K_T.cols = K->rows;

    mat_traspose(K, &K_T);
    matrix_multiply(Q, &K_T, &scores);
    mat_scaler_mult(&scores, 1.0f / sqrt(dim_k), &scores);
    mat_mask_trasform(&scores, mask);
    mat_softmax(&scores, &sof_scores);

    // Calculate the output
    matrix_multiply(&sof_scores, V, output);

    free_matrix(&scores);
    free_matrix(&sof_scores);
    free_matrix(&K_T);
    //printf("Attention output: %dx%d\n", output->rows, output->cols);
}

void attention_block(GPT2Model *model, int layer_id, Matrix *input, Matrix *output, Matrix *mask)
{
    // This function should implement the attention mechanism
    // For now, we will just print a message
    int head_dim = HIDDEN_SIZE / N_HEADS;
    //printf("Processing attention block for layer %d with head dimension %d\n", layer_id, head_dim);
    Matrix qkv;
    qkv.rows = input->rows;
    qkv.cols = HIDDEN_SIZE * 3; // Query, Key, Value
    qkv.weights = malloc_and_check(qkv.rows * qkv.cols * sizeof(float));
    matrix_multiply(input, &model->layers[layer_id].attn.attn_weight, &qkv);

    //printf(" transformer block (%d, %d) input and model->layer->attn-attn_bias (%d) \n", qkv.rows, qkv.cols, model->layers[layer_id].attn.attn_bias.cols);
    matrix_add_vector(&qkv, &model->layers[layer_id].attn.attn_bias, &qkv, 1);

    Matrix cat_attention_output;
    cat_attention_output.rows = qkv.rows;
    cat_attention_output.cols = HIDDEN_SIZE; // Concatenated output
    cat_attention_output.weights = malloc_and_check(cat_attention_output.rows * cat_attention_output.cols * sizeof(float));

    for (int i = 0; i < N_HEADS; i++)
    {
        // Split qkv into Q, K, V
        Matrix Q, K, V;
        Q.rows = qkv.rows;
        Q.cols = head_dim;
        K.rows = qkv.rows;
        K.cols = head_dim;
        V.rows = qkv.rows;
        V.cols = head_dim;

        // The output of attention for each head should have shape (seq_len, head_dim)
        // So, output.rows = Q.rows (sequence length), output.cols = head_dim
        Matrix out;
        out.rows = Q.rows;
        out.cols = head_dim;
        out.weights = malloc_and_check(out.rows * out.cols * sizeof(float));

        // Allocate memory for Q, K, V
        Q.weights = malloc_and_check(Q.rows * Q.cols * sizeof(float));
        K.weights = malloc_and_check(K.rows * K.cols * sizeof(float));
        V.weights = malloc_and_check(V.rows * V.cols * sizeof(float));

        // Extract Q, K, V from qkv
        for (int j = 0; j < qkv.rows; j++)
        {
            memcpy(Q.weights + j * head_dim, qkv.weights + j * (HIDDEN_SIZE * 3) + i * head_dim, head_dim * sizeof(float));
            memcpy(K.weights + j * head_dim, qkv.weights + j * (HIDDEN_SIZE * 3) + (i + N_HEADS) * head_dim, head_dim * sizeof(float));
            memcpy(V.weights + j * head_dim, qkv.weights + j * (HIDDEN_SIZE * 3) + (i + 2 * N_HEADS) * head_dim, head_dim * sizeof(float));
        }

        // Apply attention mechanism
        attention(&Q, &K, &V, mask, &out);

        // Copy head output to concatenated output
        for (int j = 0; j < out.rows; j++)
        {
            memcpy(cat_attention_output.weights + j * HIDDEN_SIZE + i * head_dim,
                   out.weights + j * head_dim,
                   head_dim * sizeof(float));
        }

        // Free Q, K, V, output
        free_matrix(&Q);
        free_matrix(&K);
        free_matrix(&V);
        free_matrix(&out);
    }

    matrix_multiply(&cat_attention_output, &model->layers[layer_id].attn.out_proj_weight, output);

    //printf(" transformer block project output (%d, %d) and model->layer->attn-out_proj_bias (%d) \n", output->rows, output->cols, model->layers[layer_id].attn.out_proj_bias.cols);
    matrix_add_vector(output, &model->layers[layer_id].attn.out_proj_bias, output, 2);

    free_matrix(&cat_attention_output);
    free_matrix(&qkv);
    //printf("Processed attention block layer %d\n", layer_id);
}

void mlp(GPT2Model *model, int layer_id, Matrix *input, Matrix *output)
{

    FeedForward ff = model->layers[layer_id].ff;

    // Matrix fc1_T;
    // fc1_T.weights = malloc_and_check(ff.fc1_weight.rows * ff.fc1_weight.cols * sizeof(float));
    // fc1_T.rows = ff.fc1_weight.cols;
    // fc1_T.cols = ff.fc1_weight.rows;
    // mat_traspose(&ff.fc1_weight, &fc1_T);

    Matrix hidden;
    hidden.weights = malloc_and_check(input->rows * ff.fc1_weight.cols * sizeof(float));
    hidden.rows = input->rows;
    hidden.cols = ff.fc1_weight.cols;
    matrix_multiply(input, &ff.fc1_weight, &hidden);
    //printf(" feed forward block input and ff.fc1_bias \n");
    //printf(" feed forward block input (%d, %d) and ff.fc1_bias (%d) \n", hidden.rows, hidden.cols, ff.fc1_bias.cols);
    matrix_add_vector(&hidden, &ff.fc1_bias, &hidden, 3);

    gelu(&hidden, &hidden);

    matrix_multiply(&hidden, &ff.fc2_weight, output);

    //printf(" feed forward block input (%d, %d) and ff.fc2_bias (%d) \n", output->rows, output->cols, ff.fc2_bias.cols);
    matrix_add_vector(output, &ff.fc2_bias, output, 4);

    free_matrix(&hidden);
}

void forward(GPT2Model *model, int *input, int seq_len, Matrix *logits)
{
    // float *token_embeddings = malloc_and_check(seq_len * model->token_embedding.cols * sizeof(float));
    // float *mask = malloc_and_check(seq_len * seq_len * sizeof(float));
    // float *position_embeddings = malloc_and_check(seq_len * model->position_embedding.cols * sizeof(float));
    // int *pos_input_ids = malloc_and_check(seq_len * sizeof(int));

    //printf("%d, %d\n", seq_len, model->token_embedding.cols);

    Matrix token_embeddings, position_embeddings, mask;
    token_embeddings.weights = malloc_and_check(seq_len * model->token_embedding.cols * sizeof(float));
    token_embeddings.rows = seq_len;
    token_embeddings.cols = model->token_embedding.cols;
    position_embeddings.weights = malloc_and_check(seq_len * model->position_embedding.cols * sizeof(float));
    position_embeddings.rows = seq_len;
    position_embeddings.cols = model->position_embedding.cols;
    int *pos_input_ids = malloc_and_check(seq_len * sizeof(int));

    mask.weights = malloc_and_check(seq_len * seq_len * sizeof(float));
    mask.rows = seq_len;
    mask.cols = seq_len;
    // Initialize mask to zero
    memset(mask.weights, 0, seq_len * seq_len * sizeof(float));
    mat_mask_trill(&mask);

    // Initialize position input IDs
    for (int i = 0; i < seq_len; i++)
    {
        pos_input_ids[i] = i;
    }

    //printf("%d, %d\n", model->token_embedding.rows, model->token_embedding.cols);
    // Get token embeddings
    get_embeddings(&model->token_embedding, input, seq_len, token_embeddings.weights);
    // Get position embeddings
    get_embeddings(&model->position_embedding, pos_input_ids, seq_len, position_embeddings.weights);

    // Combine token and position embeddings
    matrix_add(&token_embeddings, &position_embeddings, &token_embeddings);

    // clear unused memory
    free_matrix(&position_embeddings);
    free_ptr(&pos_input_ids);

    Matrix norm_output; // 5 x 768
    norm_output.weights = malloc_and_check(seq_len * model->token_embedding.cols * sizeof(float));
    norm_output.rows = seq_len;
    norm_output.cols = model->token_embedding.cols;

    Matrix mlp_out;
    mlp_out.weights = malloc_and_check(seq_len * model->token_embedding.cols * sizeof(float));
    mlp_out.rows = seq_len;
    mlp_out.cols = model->token_embedding.cols;

    Matrix attn_block_out; // 5 x 768
    attn_block_out.weights = malloc_and_check(seq_len * model->token_embedding.cols * sizeof(float));
    attn_block_out.rows = seq_len;
    attn_block_out.cols = model->token_embedding.cols;

    for (int i = 0; i < N_LAYERS; i++)
    {
        //printf("Processing block %d\n", i);

        // Forward pass through each transformer block
        TransformerBlock *layer = &model->layers[i];

        // float *norm_output = malloc_and_check(seq_len * model->token_embedding.cols * sizeof(float));

        memset(norm_output.weights, 0, norm_output.rows * norm_output.cols * sizeof(float));
        layer_norm(&token_embeddings, &norm_output, &layer->norm1_weight, &layer->norm1_bias);

        
        memset(attn_block_out.weights, 0, attn_block_out.rows * attn_block_out.cols * sizeof(float));
        attention_block(model, i, &norm_output, &attn_block_out, &mask);
        matrix_add(&token_embeddings, &attn_block_out, &token_embeddings);

        memset(norm_output.weights, 0, norm_output.rows * norm_output.cols * sizeof(float));
        layer_norm(&token_embeddings, &norm_output, &layer->norm2_weight, &layer->norm2_bias);

        memset(mlp_out.weights, 0, mlp_out.rows * mlp_out.cols * sizeof(float));
        mlp(model, i, &norm_output, &mlp_out);
        matrix_add(&token_embeddings, &mlp_out, &token_embeddings);

        // Apply self-attention, feed-forward, etc. (not implemented here)
        // This is where you would implement the actual transformer block logic
        // For now, we will just print the layer index

        //printf("Processed block %d\n", i);
    }

    memset(norm_output.weights, 0, norm_output.rows * norm_output.cols * sizeof(float));
    // final norm
    layer_norm(&token_embeddings, &norm_output, &model->norm_final_weight, &model->norm_final_bias);

    Matrix wte_T;
    wte_T.weights = malloc_and_check(model->token_embedding.cols * model->token_embedding.rows * sizeof(float));
    wte_T.rows = model->token_embedding.cols;
    wte_T.cols = model->token_embedding.rows;

    mat_traspose(&model->token_embedding, &wte_T);
    matrix_multiply(&norm_output, &wte_T, logits);

    // Free allocated memory
    free_matrix(&token_embeddings);
    free_matrix(&position_embeddings);
    free_ptr(&pos_input_ids);
    free_matrix(&mask);
    free_matrix(&norm_output);
    free_matrix(&mlp_out);
    free_matrix(&wte_T);
    free_matrix(&attn_block_out);
}

void generate_text(GPT2Model *model, BPETokenizer *tokenizer, char *in, char *out, int num_tokens)
{

    int input_ids[MAX_TOKENS];
    int seq_len = encode(tokenizer, in, input_ids, MAX_TOKENS);
    int max_len = seq_len + num_tokens;  

    printf("Encoded IDs: ");
    for (int i = 0; i < seq_len; i++) {
        printf("%d ", input_ids[i]);
    }
    printf("\n");


    clock_t start = clock(); 
    for (int i = seq_len; i < max_len; i++)
    {

        Matrix logits;
        logits.weights = malloc_and_check(i * VOCAB_SIZE * sizeof(float));
        logits.rows = i;
        logits.cols = VOCAB_SIZE;

        forward(model, input_ids, i, &logits);

        // Get logits for last token (last row)
        float temperature = 1.2f;
        int vocab_size = logits.cols;

        Matrix next_token_logits;
        next_token_logits.weights = malloc_and_check(vocab_size * sizeof(float));
        next_token_logits.rows = 1;
        next_token_logits.cols = vocab_size;

        //printf(" logtis %dx%d \n", logits.rows, logits.cols);
        for (int j = 0; j < vocab_size; j++)
        {
            next_token_logits.weights[j] = logits.weights[(i - 1) * vocab_size + j] / temperature;
        }

        //printf("next_token_logits (sample): ");
        // for (int j = 0; j < 10; j++)
        // {
        //     printf("%.2f ", next_token_logits.weights[j]);
        // }
        // printf("\n");

        Matrix probs;
        probs.weights = malloc_and_check(vocab_size * sizeof(float));
        probs.rows = 1;
        probs.cols = vocab_size;
        mat_softmax(&next_token_logits, &probs);

        // Top-k sampling (k=50)
        int top_k = 50;
        int *top_indices = malloc_and_check(top_k * sizeof(int));
        float *top_probs = malloc_and_check(top_k * sizeof(float));

        // Find top-k indices
        for (int k = 0; k < top_k; k++)
        {
            float max_prob = -INFINITY;
            int max_idx = -1;
            for (int j = 0; j < vocab_size; j++)
            {
                int already_selected = 0;
                for (int m = 0; m < k; m++)
                {
                    if (top_indices[m] == j)
                    {
                        already_selected = 1;
                        break;
                    }
                }
                if (!already_selected && probs.weights[j] > max_prob)
                {
                    max_prob = probs.weights[j];
                    max_idx = j;
                }
            }
            if (max_idx == -1)
            {
                fprintf(stderr, "No valid token found at top_k[%d]\n", k);
                top_indices[k] = 0;
                top_probs[k] = 0.0f;
            }
            else
            {
                top_indices[k] = max_idx;
                top_probs[k] = probs.weights[max_idx];
            }
        }

        // for (int j = 0; j < 5; j++)
        // {
        //     printf("token %d = %.4f\n", top_indices[j], top_probs[j]);
        // }

        // Renormalize top_probs
        float sum_top_probs = 0.0f;
        for (int k = 0; k < top_k; k++)
            sum_top_probs += top_probs[k];
        for (int k = 0; k < top_k; k++)
            top_probs[k] /= sum_top_probs;

        // Sample next token from top-k
        float r = (float)rand() / (float)RAND_MAX;
        float cum_prob = 0.0f;
        int next_token_id = top_indices[0];
        for (int k = 0; k < top_k; k++)
        {
            cum_prob += top_probs[k];
            if (r < cum_prob)
            {
                next_token_id = top_indices[k];
                break;
            }
        }

        input_ids[i] = next_token_id;
        if (i == seq_len)
        {
            clock_t end = clock();
            double ttft = (double)(end - start) / CLOCKS_PER_SEC;
            printf("🕒 TTFT (Time To First Token): %.4f seconds\n", ttft);
        }

        // Free memory
        free_matrix(&next_token_logits);
        free_matrix(&probs);
        free_ptr(&top_indices);
        free_ptr(&top_probs);
        free_matrix(&logits);
    }

    // print input ids
    printf("Generated sequence: ");
    for (int j = 0; j < max_len; j++)
    {
        printf("%d ", input_ids[j]);
    }

    decode(tokenizer, input_ids, max_len, out, MAX_OUTPUT_LEN);
    printf("\n");
}

int load_weights(GPT2Model *model)
{
    // Load weights for each layer, token embedding, position embedding, etc.
    // This function should load all necessary weights from files
    // For now, we will just print a message
    printf("Loading model weights...\n");
    load_matrix("gpt2_weights/token_embedding.bin", &model->token_embedding);
    load_matrix("gpt2_weights/position_embedding.bin", &model->position_embedding);
    for (int i = 0; i < N_LAYERS; i++)
    {
        char filename[256];
        // snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d.bin", i);
        //  Load each layer's weights (not implemented here)
        //  You would need to implement the logic to load each layer's weights
        // printf("Loading layer %d weights from %s\n", i, filename);
        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_ln1_weight.bin", i);
        //printf("Loading layer %d weights from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].norm1_weight);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_ln2_weight.bin", i);
        //printf("Loading layer %d weights from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].norm2_weight);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_ln1_bias.bin", i);
        //printf("Loading layer %d weights from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].norm1_bias);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_ln2_bias.bin", i);
        //printf("Loading layer %d weights from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].norm2_bias);

        // feed forward weights and biases
        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_mlp_fc_weight.bin", i);
        //printf("Loading layer %d feed forward weights from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].ff.fc1_weight);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_mlp_fc_bias.bin", i);
        //printf("Loading layer %d feed forward bias from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].ff.fc1_bias);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_mlp_proj_weight.bin", i);
        //printf("Loading layer %d feed forward weights from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].ff.fc2_weight);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_mlp_proj_bias.bin", i);
        //printf("Loading layer %d feed forward bias from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].ff.fc2_bias);

        // attention weights, biases, projection weights and biases
        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_attn_c_attn_weight.bin", i);
        //printf("Loading layer %d attention weights from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].attn.attn_weight);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_attn_c_attn_bias.bin", i);
        //printf("Loading layer %d attention bias from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].attn.attn_bias);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_attn_c_proj_weight.bin", i);
        //printf("Loading layer %d attention projection weights from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].attn.out_proj_weight);

        snprintf(filename, sizeof(filename), "gpt2_weights/layer_%d_attn_c_proj_bias.bin", i);
        //printf("Loading layer %d attention projection bias from %s\n", i, filename);
        load_matrix(filename, &model->layers[i].attn.out_proj_bias);
    }

    // Load final layer norm weights and biases
    load_matrix("gpt2_weights/final_ln_weight.bin", &model->norm_final_weight);
    load_matrix("gpt2_weights/final_ln_bias.bin", &model->norm_final_bias);

    return 0; // Return 0 on success
}

int free_matrix(Matrix *matrix)
{
    if (matrix != NULL && matrix->weights != NULL)
    {
        free_ptr(&matrix->weights);
        matrix->rows = 0;
        matrix->cols = 0;
        matrix->weights = NULL;
        return 0; // Return 0 on success
    }
    matrix->rows = 0;
    matrix->cols = 0;
    return -1; // Return -1 if matrix is NULL
}

int free_gpt2_model(GPT2Model *model)
{
    if (model != NULL)
    {
        for (int i = 0; i < N_LAYERS; i++)
        {
            free_matrix(&model->layers[i].attn.attn_weight);
            free_matrix(&model->layers[i].attn.attn_bias);
            free_matrix(&model->layers[i].attn.out_proj_weight);
            free_matrix(&model->layers[i].attn.out_proj_bias);
            free_matrix(&model->layers[i].ff.fc1_weight);
            free_matrix(&model->layers[i].ff.fc1_bias);
            free_matrix(&model->layers[i].ff.fc2_weight);
            free_matrix(&model->layers[i].ff.fc2_bias);
            free_matrix(&model->layers[i].norm1_weight);
            free_matrix(&model->layers[i].norm1_bias);
            free_matrix(&model->layers[i].norm2_weight);
            free_matrix(&model->layers[i].norm2_bias);
        }
        free_matrix(&model->token_embedding);
        free_matrix(&model->position_embedding);
        free_matrix(&model->norm_final_weight);
        free_matrix(&model->norm_final_bias);

        return 0; // Return 0 on success
    }
    return -1; // Return -1 if model is NULL
}

int main(int argc, char *argv[])
{

    if (argc < 3) {
                fprintf(stderr, "Usage: %s <num_token_to_generate> <input_string> \n", argv[0]);
        return EXIT_FAILURE;
    }

    GPT2Model model;
    memset(&model, 0, sizeof(GPT2Model));
    srand(time(NULL));

    //int inp_ids[] = {1169, 37443, 1659, 220, 32683}; // Example input IDs
    int inp_ids[] = {72, 321, 1169, 3364, 1659};
    int seq_len = sizeof(inp_ids) / sizeof(inp_ids[0]);
    int num_seq = atoi(argv[1]);
    const char *input = argv[2];
   

    BPETokenizer tokenizer = {0};
    tokenizer.vocab = malloc(MAX_TOKENS * MAX_TOKEN_LEN);
    // Allocate merges (e.g., 50,000)
    tokenizer.merges = malloc(MAX_MERGES * sizeof(*tokenizer.merges));

    if (!tokenizer.vocab || !tokenizer.merges) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }


    load_vocab(&tokenizer, "gpt2_cache/vocab.txt");
    load_merges(&tokenizer, "gpt2_cache/merges.txt");


    /*
    * load mode 
    */
    if (load_weights(&model) != 0)
    {
        fprintf(stderr, "Failed to load model weights\n");
        return EXIT_FAILURE; // Exit if loading weights fails
    }

    printf("Model weights loaded successfully\n");
    int num_params = matrix_num_params(&model);

    printf("Model Num Params: %0.2fM\n", num_params/1e6f);
    printf("Model Size: %0.2f MB\n", num_params * sizeof(float) / (1024.0 * 1024.0));
    // Free allocated memory for model


    char out[MAX_OUTPUT_LEN];
    clock_t start = clock();
    generate_text(&model, &tokenizer, input, out, num_seq);

    clock_t end = clock(); // End timer
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total generation time: %.4f seconds\n", time_taken); 
    printf("Decoded: %s\n", out);


    // free_gpt2_model(&model);
    // free_ptr(&input_ids);

    free_ptr(&tokenizer.vocab);
    free_ptr(&tokenizer.merges);
    return EXIT_SUCCESS;
}
