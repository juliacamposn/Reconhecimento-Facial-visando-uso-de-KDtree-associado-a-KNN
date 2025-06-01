#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>  
#include <float.h>
#include "kdtree.h" 

/* --- Variável Global da Árvore --- */
static KDTree global_kd_tree;
static int tree_initialized = 0;

/* --- Implementação do Max-Heap --- */

MaxHeap* create_max_heap(int capacity) {
    MaxHeap *heap = (MaxHeap*)malloc(sizeof(MaxHeap));
    if (!heap) return NULL;
    heap->nodes = (HeapNode*)malloc(capacity * sizeof(HeapNode));
    if (!heap->nodes) {
        free(heap);
        return NULL;
    }
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void destroy_max_heap(MaxHeap *heap) {
    if (heap) {
        free(heap->nodes);
        free(heap);
    }
}

void swap_heap_nodes(HeapNode *a, HeapNode *b) {
    HeapNode temp = *a;
    *a = *b;
    *b = temp;
}

void max_heapify_up(MaxHeap *heap, int index) {
    if (index && heap->nodes[index].distance > heap->nodes[(index - 1) / 2].distance) {
        swap_heap_nodes(&heap->nodes[index], &heap->nodes[(index - 1) / 2]);
        max_heapify_up(heap, (index - 1) / 2);
    }
}

void max_heapify_down(MaxHeap *heap, int index) {
    int largest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;

    if (left < heap->size && heap->nodes[left].distance > heap->nodes[largest].distance)
        largest = left;
    if (right < heap->size && heap->nodes[right].distance > heap->nodes[largest].distance)
        largest = right;

    if (largest != index) {
        swap_heap_nodes(&heap->nodes[index], &heap->nodes[largest]);
        max_heapify_down(heap, largest);
    }
}

void heap_push(MaxHeap *heap, FaceRecord *record, double distance) {
    if (heap->size < heap->capacity) {
        heap->nodes[heap->size].record = record;
        heap->nodes[heap->size].distance = distance;
        heap->size++;
        max_heapify_up(heap, heap->size - 1);
    } else if (distance < heap->nodes[0].distance) { // Se a nova distância é menor que a maior no max-heap
        heap->nodes[0].record = record;
        heap->nodes[0].distance = distance;
        max_heapify_down(heap, 0);
    }
}

HeapNode heap_pop_max(MaxHeap *heap) { // Não estritamente necessário para k-NN, mas bom ter
    if (heap->size <= 0) return (HeapNode){NULL, -1.0}; // Erro ou heap vazio
    HeapNode max_node = heap->nodes[0];
    heap->nodes[0] = heap->nodes[heap->size - 1];
    heap->size--;
    max_heapify_down(heap, 0);
    return max_node;
}

/* --- Funções da KD-Tree --- */

// Função de distância (Euclidiana ao quadrado para evitar sqrt até o final)
double calculate_distance_sq(const FaceRecord *r1, const FaceRecord *r2) {
    double dist_sq = 0.0;
    for (int i = 0; i < K_DIMENSIONS; ++i) {
        double diff = r1->embedding[i] - r2->embedding[i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

// Aloca e inicializa um novo FaceRecord (copia os dados)
FaceRecord* create_face_record(const float embedding[K_DIMENSIONS], const char person_id[MAX_ID_LENGTH]) {
    FaceRecord *new_record = (FaceRecord*)malloc(sizeof(FaceRecord));
    if (!new_record) {
        perror("Failed to allocate FaceRecord");
        return NULL;
    }
    memcpy(new_record->embedding, embedding, K_DIMENSIONS * sizeof(float));
    strncpy(new_record->person_id, person_id, MAX_ID_LENGTH - 1);
    new_record->person_id[MAX_ID_LENGTH - 1] = '\0'; // Garante terminação nula
    new_record->distance_to_query = 0.0; // Inicializa
    return new_record;
}


KDNode* _kdtree_insert_recursive(KDNode *current, FaceRecord *record_data, int depth) {
    if (current == NULL) {
        KDNode *newNode = (KDNode*)malloc(sizeof(KDNode));
        if (!newNode) {
            perror("Failed to allocate KDNode");
            // O record_data foi alocado antes, precisa ser liberado se a inserção falhar aqui
            // No entanto, a kdtree_insert_global já lida com isso.
            return NULL;
        }
        newNode->data = record_data; // Assume que record_data já foi alocado
        newNode->left = newNode->right = NULL;
        return newNode;
    }

    int axis = depth % global_kd_tree.k;

    if (record_data->embedding[axis] < current->data->embedding[axis]) {
        current->left = _kdtree_insert_recursive(current->left, record_data, depth + 1);
    } else {
        current->right = _kdtree_insert_recursive(current->right, record_data, depth + 1);
    }
    return current;
}

void kdtree_insert_global(FaceRecord *record_to_insert) {
    if (!tree_initialized) {
        fprintf(stderr, "Error: KD-Tree not initialized.\n");
        // Libera o registro que não será inserido, pois quem chama espera que a árvore tome posse
        free(record_to_insert);
        return;
    }
    if (!record_to_insert) {
        fprintf(stderr, "Error: Cannot insert NULL record.\n");
        return;
    }
    global_kd_tree.root = _kdtree_insert_recursive(global_kd_tree.root, record_to_insert, 0);
}


void _kdtree_knn_recursive(KDNode *current, const FaceRecord *query_record, int depth, MaxHeap *knn_heap) {
    if (current == NULL) {
        return;
    }

    int axis = depth % global_kd_tree.k;
    double dist_sq = calculate_distance_sq(current->data, query_record);

    heap_push(knn_heap, current->data, dist_sq);

    KDNode *near_child, *far_child;
    if (query_record->embedding[axis] < current->data->embedding[axis]) {
        near_child = current->left;
        far_child = current->right;
    } else {
        near_child = current->right;
        far_child = current->left;
    }

    _kdtree_knn_recursive(near_child, query_record, depth + 1, knn_heap);

    // Verifica se precisa buscar no lado oposto (poda)
    double axis_dist_sq = query_record->embedding[axis] - current->data->embedding[axis];
    axis_dist_sq *= axis_dist_sq;

    // Se o heap não está cheio ou a hiperesfera de busca cruza o hiperplano divisor
    if (knn_heap->size < knn_heap->capacity || axis_dist_sq < knn_heap->nodes[0].distance) {
        _kdtree_knn_recursive(far_child, query_record, depth + 1, knn_heap);
    }
}

// Compara HeapNodes para qsort (ordem crescente de distância)
int compare_heap_nodes(const void *a, const void *b) {
    HeapNode *nodeA = (HeapNode *)a;
    HeapNode *nodeB = (HeapNode *)b;
    if (nodeA->distance < nodeB->distance) return -1;
    if (nodeA->distance > nodeB->distance) return 1;
    return 0;
}

int kdtree_find_n_nearest_global(const FaceRecord *query_record, int n_neighbors, FaceRecord *found_neighbors) {
    if (!tree_initialized || global_kd_tree.root == NULL) {
        fprintf(stderr, "KD-Tree is empty or not initialized.\n");
        return 0;
    }
    if (n_neighbors <= 0) return 0;

    MaxHeap *knn_heap = create_max_heap(n_neighbors);
    if (!knn_heap) {
        perror("Failed to create MaxHeap for k-NN search");
        return 0;
    }

    _kdtree_knn_recursive(global_kd_tree.root, query_record, 0, knn_heap);

    // Extrair os resultados do heap e ordená-los (o heap já os mantém, mas não estritamente ordenados)
    // O heap contém os N mais próximos, com o mais distante deles no topo (heap_nodes[0]).
    // Para retornar em ordem de proximidade, precisamos extrair e ordenar.
    int num_found = knn_heap->size;
    for (int i = 0; i < num_found; ++i) {
        // Copia os dados para a estrutura de saída
        // Atenção: found_neighbors deve ter sido alocado pelo chamador com tamanho suficiente
        memcpy(found_neighbors[i].embedding, knn_heap->nodes[i].record->embedding, K_DIMENSIONS * sizeof(float));
        strncpy(found_neighbors[i].person_id, knn_heap->nodes[i].record->person_id, MAX_ID_LENGTH -1);
        found_neighbors[i].person_id[MAX_ID_LENGTH-1] = '\0';
        found_neighbors[i].distance_to_query = sqrt(knn_heap->nodes[i].distance); // Armazena a distância real
    }
    
    // Ordena os resultados pela distância
    qsort(found_neighbors, num_found, sizeof(FaceRecord), compare_heap_nodes);


    destroy_max_heap(knn_heap);
    return num_found;
}


void kdtree_initialize_global() {
    global_kd_tree.root = NULL;
    global_kd_tree.k = K_DIMENSIONS;
    tree_initialized = 1;
    printf("Global KD-Tree initialized with K=%d.\n", K_DIMENSIONS);
}

void _kdtree_destroy_recursive(KDNode *node) {
    if (node) {
        _kdtree_destroy_recursive(node->left);
        _kdtree_destroy_recursive(node->right);
        free(node->data); // Libera o FaceRecord
        free(node);       // Libera o KDNode
    }
}

void kdtree_destroy_global() {
    if (tree_initialized) {
        _kdtree_destroy_recursive(global_kd_tree.root);
        global_kd_tree.root = NULL;
        tree_initialized = 0;
        printf("Global KD-Tree destroyed.\n");
    }
}

KDTree* get_global_kdtree() {
    if (!tree_initialized) {
         fprintf(stderr, "Warning: Accessing uninitialized global KD-Tree. Initializing now.\n");
         kdtree_initialize_global();
    }
    return &global_kd_tree;
}

// Funções que serão exportadas pela DLL/SO para o Python (wrappers)

// Função para ser chamada pelo Python para construir a árvore (inicializar)
void EXPORT_kdtree_construir() {
    if (tree_initialized) {
        kdtree_destroy_global(); // Destrói se já existir para recomeçar
    }
    kdtree_initialize_global();
}

// Função para ser chamada pelo Python para inserir um ponto
// O Python passará uma estrutura FaceRecord por valor (ou ponteiro)
void EXPORT_inserir_ponto(FaceRecord record_from_python) {
    if (!tree_initialized) {
        fprintf(stderr, "Error: KD-Tree not initialized before insert. Call kdtree_construir first.\n");
        return;
    }
    // Precisamos alocar memória para este registro no lado C, pois a árvore vai armazenar ponteiros
    FaceRecord *new_record = create_face_record(record_from_python.embedding, record_from_python.person_id);
    if (new_record) {
        kdtree_insert_global(new_record);
    } else {
        fprintf(stderr, "Error: Failed to create FaceRecord for insertion.\n");
    }
}

// Função para ser chamada pelo Python para buscar N vizinhos
// Python passará: query_record, n_neighbors, array_para_resultados
// Retorna o número de vizinhos efetivamente encontrados e preenchidos em 'output_records'
int EXPORT_buscar_n_vizinhos(FaceRecord query_record, int n_neighbors, FaceRecord* output_records) {
    if (!tree_initialized) {
        fprintf(stderr, "Error: KD-Tree not initialized before search.\n");
        return 0;
    }
    return kdtree_find_n_nearest_global(&query_record, n_neighbors, output_records);
}

// Opcional: Função para obter ponteiro para a árvore (para depuração ou funções avançadas)
// Normalmente, você não exporia a estrutura interna diretamente.
// KDTree* EXPORT_get_tree() {
//     return get_global_kdtree();
// }

/*
// Exemplo de main para teste (opcional, remova ou comente ao compilar como biblioteca)
int main() {
    EXPORT_kdtree_construir();

    FaceRecord fr1, fr2, fr3, query;

    for(int i=0; i<K_DIMENSIONS; ++i) fr1.embedding[i] = (float)i;
    strcpy(fr1.person_id, "Person_1");
    EXPORT_inserir_ponto(fr1);

    for(int i=0; i<K_DIMENSIONS; ++i) fr2.embedding[i] = (float)i + 0.5f;
    strcpy(fr2.person_id, "Person_2");
    EXPORT_inserir_ponto(fr2);
    
    for(int i=0; i<K_DIMENSIONS; ++i) fr3.embedding[i] = (float)i * 2.0f;
    strcpy(fr3.person_id, "Person_3");
    EXPORT_inserir_ponto(fr3);

    // Query
    for(int i=0; i<K_DIMENSIONS; ++i) query.embedding[i] = (float)i + 0.1f;
    strcpy(query.person_id, "Query_Point"); // ID da query não é usado na busca em si

    int n_vizinhos = 2;
    FaceRecord results[n_vizinhos];

    printf("\nBuscando %d vizinhos mais proximos de Query_Point...\n", n_vizinhos);
    int found_count = EXPORT_buscar_n_vizinhos(query, n_vizinhos, results);

    printf("Encontrados %d vizinhos:\n", found_count);
    for (int i = 0; i < found_count; ++i) {
        printf("  ID: %s, Distancia: %f\n", results[i].person_id, results[i].distance_to_query);
        // printf("  Embedding[0]: %f\n", results[i].embedding[0]);
    }

    kdtree_destroy_global();
    return 0;
}
*/