#ifndef KDTREE_H
#define KDTREE_H

#include <float.h> 

#define K_DIMENSIONS 128
#define MAX_ID_LENGTH 100

// Estrutura para os dados do usuário (embedding facial e ID)
typedef struct _face_record {
    float embedding[K_DIMENSIONS];
    char person_id[MAX_ID_LENGTH];
    double distance_to_query; 
} FaceRecord;

// Nó da KD-Tree
typedef struct _kd_node {
    FaceRecord *data;
    struct _kd_node *left;
    struct _kd_node *right;
} KDNode;

typedef struct _kd_tree {
    KDNode *root;
    int k; 
} KDTree;

// Estrutura para um item no Heap (usado para k-NN)
typedef struct _heap_node {
    FaceRecord *record;
    double distance;    
} HeapNode;

// Estrutura do Max-Heap
typedef struct _max_heap {
    HeapNode *nodes;
    int size;
    int capacity;
} MaxHeap;

/* --- Funções da KD-Tree --- */
// Constrói/inicializa a árvore KD global
void kdtree_initialize_global();

// Insere um novo registro de face na árvore KD global
void kdtree_insert_global(FaceRecord *record_data);

// Encontra os N vizinhos mais próximos na árvore KD global
// Retorna o número de vizinhos encontrados (pode ser < n_neighbors)
// Os resultados são preenchidos em 'found_neighbors'
int kdtree_find_n_nearest_global(const FaceRecord *query_record, int n_neighbors, FaceRecord *found_neighbors);

// Libera a memória da árvore KD global
void kdtree_destroy_global();

KDTree* get_global_kdtree();

#endif 