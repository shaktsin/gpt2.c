#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 50257
#define MAX_SEQ_LEN 1024
#define N_LAYERS 12
#define N_HEADS 12
#define HIDDEN_SIZE 768

#define MAX_TOKENS 50257
#define MAX_TOKEN_LEN 64
#define MAX_MERGES 50000
#define MAX_OUTPUT_LEN 256

typedef struct {
    //char vocab[MAX_TOKENS][MAX_TOKEN_LEN];

    char (*vocab)[MAX_TOKEN_LEN];
    int vocab_size;

    struct {
        char pair[2][MAX_TOKEN_LEN];
    } *merges;

    int merge_size;
} BPETokenizer;

void load_vocab(BPETokenizer *tokenizer, const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("Failed to open vocab file");
        exit(1);
    }
    while (tokenizer->vocab_size < MAX_TOKENS &&
           fscanf(fp, "%31s", tokenizer->vocab[tokenizer->vocab_size]) == 1)
    {
        tokenizer->vocab_size++;
    }
    fclose(fp);
}

void load_merges(BPETokenizer *tokenizer, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("Failed to open vocab file");
        exit(1);
    }
    while (fscanf(fp, "%s %s",
                  tokenizer->merges[tokenizer->merge_size].pair[0],
                  tokenizer->merges[tokenizer->merge_size].pair[1]) == 2) {
        tokenizer->merge_size++;
    }
    fclose(fp);
}

int encode(BPETokenizer *tokenizer, const char *word, int *output_ids, int max_ids) {
    char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    int len = 0;

    for (int i = 0; i < strlen(word); i++) {
        snprintf(tokens[len], MAX_TOKEN_LEN, "%c", word[i]);
        len++;
    }

    int merged = 1;
    while (merged) {
        merged = 0;
        for (int m = 0; m < tokenizer->merge_size; m++) {
            for (int i = 0; i < len - 1; i++) {
                if (strcmp(tokens[i], tokenizer->merges[m].pair[0]) == 0 &&
                    strcmp(tokens[i + 1], tokenizer->merges[m].pair[1]) == 0) {

                    char merged_token[MAX_TOKEN_LEN * 2];
                    snprintf(merged_token, sizeof(merged_token), "%s%s", tokens[i], tokens[i + 1]);

                    // Shift tokens
                    for (int j = i + 1; j < len - 1; j++) {
                        strcpy(tokens[j], tokens[j + 1]);
                    }
                    strcpy(tokens[i], merged_token);
                    len--;
                    merged = 1;
                    break;
                }
            }
            if (merged) break;
        }
    }

    int out_len = 0;
    for (int i = 0; i < len && out_len < max_ids; i++) {
        for (int j = 0; j < tokenizer->vocab_size; j++) {
            if (strcmp(tokenizer->vocab[j], tokens[i]) == 0) {
                output_ids[out_len++] = j;
                break;
            }
        }
    }

    return out_len;
}



void decode(BPETokenizer *tokenizer, const int *token_ids, int len, char *output, int max_output_len) {
    output[0] = '\0';

    for (int i = 0; i < len; i++) {
        if (token_ids[i] < 0 || token_ids[i] >= tokenizer->vocab_size) continue;
        
        char *token = tokenizer->vocab[token_ids[i]];

        // If it starts with the BPE whitespace symbol (Ġ), replace it with ' '
        if (token[0] == '\u0120' || token[0] == 0xC4) { // Ġ in UTF-8 is 0xC4 0xA0
            strncat(output, " ", max_output_len - strlen(output) - 1);
            strncat(output, token + 2, max_output_len - strlen(output) - 1);  // skip Ġ
        } else {
            strncat(output, token, max_output_len - strlen(output) - 1);
        }
    }
}





typedef struct {
    float *weights;
    int rows;
    int cols;
} Matrix;

typedef struct {
    Matrix attn_weight, out_proj_weight;
    Matrix attn_bias, out_proj_bias;
} Attention;

typedef struct {
    Matrix fc1_weight, fc1_bias;
    Matrix fc2_weight, fc2_bias;
} FeedForward;

typedef struct {
    Attention attn;
    FeedForward ff;
    Matrix norm1_weight, norm1_bias;
    Matrix norm2_weight, norm2_bias;
} TransformerBlock;

typedef struct {
    TransformerBlock layers[N_LAYERS];
    Matrix token_embedding;  // [VOCAB_SIZE][HIDDEN_SIZE]
    Matrix position_embedding; // [MAX_SEQ_LEN][HIDDEN_SIZE]
    Matrix norm_final_weight, norm_final_bias;
    Matrix lm_head;  // final linear projection to vocab
} GPT2Model;

size_t matrix_num_params(const Matrix *m) {
    return (size_t)(m->rows) * (size_t)(m->cols);
}

size_t model_num_params(const GPT2Model *model) {
    size_t total = 0;

    // Token and position embeddings
    total += matrix_num_params(&model->token_embedding);
    total += matrix_num_params(&model->position_embedding);

    // Final normalization and lm_head
    total += matrix_num_params(&model->norm_final_weight);
    total += matrix_num_params(&model->norm_final_bias);
    total += matrix_num_params(&model->lm_head);

    // Layers
    for (int i = 0; i < N_LAYERS; i++) {
        const TransformerBlock *layer = &model->layers[i];

        // Attention weights
        total += matrix_num_params(&layer->attn.attn_weight);
        total += matrix_num_params(&layer->attn.out_proj_weight);
        total += matrix_num_params(&layer->attn.attn_bias);
        total += matrix_num_params(&layer->attn.out_proj_bias);

        // Feedforward weights
        total += matrix_num_params(&layer->ff.fc1_weight);
        total += matrix_num_params(&layer->ff.fc1_bias);
        total += matrix_num_params(&layer->ff.fc2_weight);
        total += matrix_num_params(&layer->ff.fc2_bias);

        // LayerNorm weights
        total += matrix_num_params(&layer->norm1_weight);
        total += matrix_num_params(&layer->norm1_bias);
        total += matrix_num_params(&layer->norm2_weight);
        total += matrix_num_params(&layer->norm2_bias);
    }

    return total;
}

int free_ptr(void **ptr) {
    if (ptr != NULL && *ptr != NULL) {
        free(*ptr);
        *ptr = NULL; // Set pointer to NULL after freeing
        return 0; // Return 0 on success
    }
    return -1; // Return -1 if ptr is NULL
}

void *malloc_and_check(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE); // Exit on memory allocation failure
    }
    return ptr;
}

int matrix_multiply(const Matrix *a, const Matrix *b, Matrix *result) {
    if (a->cols != b->rows || result->rows != a->rows || result->cols != b->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication\n");
        return -1; // Return -1 on dimension mismatch
    }

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            result->weights[i * result->cols + j] = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                result->weights[i * result->cols + j] += a->weights[i * a->cols + k] * b->weights[k * b->cols + j];
            }
        }
    }
    return 0; // Return 0 on success
}

int mat_scaler_mult(const Matrix *input, float scalar, Matrix *output) {
    if (input->rows != output->rows || input->cols != output->cols) {
        fprintf(stderr, "Matrix dimensions do not match for scalar multiplication\n");
        return -1; // Return -1 on dimension mismatch
    }

    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            output->weights[i * output->cols + j] = input->weights[i * input->cols + j] * scalar;
        }
    }
    return 0; // Return 0 on success
}

int mat_mask_trill(Matrix *input) {
    // This function should implement the masking logic
    // For now, we will just print a message
    //printf("Applying mask to matrix with dimensions %dx%d\n", input->rows, input->cols);
    
    // Example: Set all elements to zero (this is just a placeholder)
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {

            if (j <= i)
            {
                input->weights[i * input->cols + j] = 1.0f; // allow
            }
            else
            {
                input->weights[i * input->cols + j] = 0.0f; // mask
            }
        }
    }
    return 0; // Return 0 on success
}

int mat_mask_trasform(Matrix *input, Matrix *mask) {
    // This function should implement the masking logic for transformer
    // For now, we will just print a message
    //printf("Applying transformer mask to matrix with dimensions %dx%d\n", input->rows, input->cols);
    
    // Example: Set all elements to zero (this is just a placeholder)
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            if (mask->weights[i * mask->cols + j] == 0.0f) {
                input->weights[i * input->cols + j] = -1e9; // Set masked elements to -1e9
            }
        }
    }
    return 0; // Return 0 on success
}

int mat_softmax(const Matrix *input, Matrix *output) {
    if (input->rows != output->rows || input->cols != output->cols) {
        fprintf(stderr, "Matrix dimensions do not match for softmax\n");
        return -1; // Return -1 on dimension mismatch
    }

    for (int i = 0; i < input->rows; i++) {
        float max_val = -1e9f;
        float sum = 0.0f;

        // Find max value in the row
        for (int j = 0; j < input->cols; j++) {
            if (input->weights[i * input->cols + j] > max_val) {
                max_val = input->weights[i * input->cols + j];
            }
        }

        // Calculate softmax
        for (int j = 0; j < input->cols; j++) {
            output->weights[i * output->cols + j] = expf(input->weights[i * input->cols + j] - max_val);
            sum += output->weights[i * output->cols + j];
        }

        // Normalize
        for (int j = 0; j < input->cols; j++) {
            output->weights[i * output->cols + j] /= sum;
        }
    }
    return 0; // Return 0 on success
}


int matrix_add(const Matrix *a, const Matrix *b, Matrix *result) {
    if (a->rows != b->rows || a->cols != b->cols || result->rows != a->rows || result->cols != a->cols) {
        fprintf(stderr, "Matrix dimensions do not match for addition\n");
        return -1; // Return -1 on dimension mismatch
    }

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->weights[i * result->cols + j] = a->weights[i * a->cols + j] + b->weights[i * b->cols + j];
        }
    }
    return 0; // Return 0 on success
}

// b is a vector
int matrix_add_vector(const Matrix *a, const Matrix *b, Matrix *result, int idx) {

    if (a->cols != b->cols) {
        fprintf(stderr, "(%d) - Matrix dimensions do not match for vector addition a, b - (%d, %d) \n", idx, a->cols, b->cols);
        return -1; // Return -1 on dimension mismatch
    }

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->weights[i * result->cols + j] = a->weights[i * a->cols + j] + b->weights[j];
        }
    }
    return 0; // Return 0 on success
}

int mat_traspose(const Matrix *input, Matrix *output) {
    if (input->rows != output->cols || input->cols != output->rows) {
        fprintf(stderr, "Matrix dimensions do not match for transpose\n");
        return -1; // Return -1 on dimension mismatch
    }

    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            output->weights[j * output->cols + i] = input->weights[i * input->cols + j];
        }
    }
    return 0; // Return 0 on success
}

int gelu(const Matrix *input, Matrix *output) {
    if (input->rows != output->rows || input->cols != output->cols) {
        fprintf(stderr, "Matrix dimensions do not match for GELU\n");
        return -1;
    }
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            float x = input->weights[i * input->cols + j];
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            float c = 0.044715f;
            float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
            float x3 = x * x * x;
            float tanh_arg = sqrt_2_over_pi * (x + c * x3);
            float gelu_val = 0.5f * x * (1.0f + tanhf(tanh_arg));
            output->weights[i * output->cols + j] = gelu_val;
        }
    }
    return 0;
}

int layer_norm(const Matrix *x, Matrix *output, const Matrix *weight, const Matrix *bias) {

    float eps = 1e-5; 
    if (x->rows != output->rows || x->cols != output->cols ||
        weight->rows != 1 || weight->cols != x->cols ||
        bias->rows != 1 || bias->cols != x->cols) {
        fprintf(stderr, "Matrix dimensions do not match for layer_norm\n");
        return -1;
    }

    for (int i = 0; i < x->rows; i++) {
        // Compute mean
        float mean = 0.0f;
        for (int j = 0; j < x->cols; j++) {
            mean += x->weights[i * x->cols + j];
        }
        mean /= x->cols;

        // Compute variance
        float var = 0.0f;
        for (int j = 0; j < x->cols; j++) {
            float diff = x->weights[i * x->cols + j] - mean;
            var += diff * diff;
        }
        var /= x->cols;

        // Normalize, scale and shift
        for (int j = 0; j < x->cols; j++) {
            float norm = (x->weights[i * x->cols + j] - mean) / sqrtf(var + eps);
            output->weights[i * output->cols + j] = weight->weights[j] * norm + bias->weights[j];
        }
    }
    return 0;
}