import ctypes
from ctypes import Structure, POINTER, c_float, c_char, c_int, c_double, c_void_p

K_DIMENSIONS = 128
MAX_ID_LENGTH = 100

class FaceRecord(Structure):
    _fields_ = [
        ("embedding", c_float * K_DIMENSIONS),
        ("person_id", c_char * MAX_ID_LENGTH),
        ("distance_to_query", c_double) # Para receber a distância do C
    ]

# As definições de KDNode e KDTree não são estritamente necessárias no Python
# se você apenas interage através das funções exportadas que usam FaceRecord.
# No entanto, se EXPORT_get_tree retornasse um POINTER(KDTree), você precisaria delas.

# Carregar a biblioteca C
# Certifique-se que libkdtree.so (ou .dll no Windows) está no caminho
try:
    lib = ctypes.CDLL("./libkdtree.so") # Linux/Outros
except OSError:
    try:
        lib = ctypes.CDLL("./libkdtree.dll") # Windows
    except OSError as e:
        print(f"Erro ao carregar a biblioteca C: {e}")
        print("Certifique-se que 'libkdtree.so' ou 'libkdtree.dll' está compilada e no mesmo diretório.")
        exit(1)


# --- Definir assinaturas das funções C exportadas ---

# void EXPORT_kdtree_construir()
lib.EXPORT_kdtree_construir.argtypes = []
lib.EXPORT_kdtree_construir.restype = None

# void EXPORT_inserir_ponto(FaceRecord record_from_python)
lib.EXPORT_inserir_ponto.argtypes = [FaceRecord] # Passa a estrutura por valor (C fará a cópia)
lib.EXPORT_inserir_ponto.restype = None

# int EXPORT_buscar_n_vizinhos(FaceRecord query_record, int n_neighbors, FaceRecord* output_records)
lib.EXPORT_buscar_n_vizinhos.argtypes = [FaceRecord, c_int, POINTER(FaceRecord)]
lib.EXPORT_buscar_n_vizinhos.restype = c_int

# Exemplo de como obter o ponteiro da árvore, se necessário (não usado no app.py atual)
# class KDNode(Structure):
#     pass # Definição recursiva precisa de um truque
# KDNode._fields_ = [("data", POINTER(FaceRecord)), # Ou c_void_p se for genérico
#                    ("left", POINTER(KDNode)),
#                    ("right", POINTER(KDNode))]

# class KDTree(Structure):
#     _fields_ = [("root", POINTER(KDNode)),
#                 ("k", c_int)]

# lib.EXPORT_get_tree.restype = POINTER(KDTree) # Se você tivesse essa função