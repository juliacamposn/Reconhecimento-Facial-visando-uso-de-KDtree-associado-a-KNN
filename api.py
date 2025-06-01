from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import ctypes

# Importa as definições e a lib do wrapper
from kdtree_wrapper import lib, FaceRecord, K_DIMENSIONS, MAX_ID_LENGTH

app = FastAPI(
    title="Face Recognition KD-Tree API",
    description="API para reconhecimento facial usando KD-Tree com embeddings."
)

class FaceDataInput(BaseModel):
    embedding: List[float] = Field(..., min_length=K_DIMENSIONS, max_length=K_DIMENSIONS)
    person_id: str = Field(..., max_length=MAX_ID_LENGTH -1) # -1 para garantir espaço para \0

class NeighborResult(BaseModel):
    person_id: str
    embedding: List[float]
    distance: float

@app.post("/initialize-tree", summary="Initialize or Re-initialize KD-Tree")
def initialize_tree_endpoint():
    """
    Initializes (or re-initializes by destroying the old one) the global KD-Tree in the C library.
    Should be called before any insertions or searches if the application restarts.
    """
    try:
        lib.EXPORT_kdtree_construir()
        return {"message": "KD-Tree initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize KD-Tree: {str(e)}")

@app.post("/insert-face", summary="Insert Face Embedding")
def insert_face_endpoint(face_data: FaceDataInput):
    """
    Inserts a face embedding and its associated person ID into the KD-Tree.
    """
    if len(face_data.embedding) != K_DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"Embedding must have {K_DIMENSIONS} dimensions.")
    if not face_data.person_id:
        raise HTTPException(status_code=400, detail="Person ID cannot be empty.")

    # Converter List[float] para (c_float * K_DIMENSIONS)
    embedding_c = (ctypes.c_float * K_DIMENSIONS)(*face_data.embedding)
    
    # Converter person_id str para bytes e depois para c_char array
    person_id_bytes = face_data.person_id.encode('utf-8')
    if len(person_id_bytes) >= MAX_ID_LENGTH:
         raise HTTPException(status_code=400, detail=f"Person ID too long (max {MAX_ID_LENGTH-1} bytes).")

    c_record = FaceRecord()
    ctypes.memmove(c_record.embedding, embedding_c, ctypes.sizeof(embedding_c))
    c_record.person_id = person_id_bytes 

    try:
        lib.EXPORT_inserir_ponto(c_record) 
        return {"message": f"Face embedding for '{face_data.person_id}' inserted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert face embedding: {str(e)}")


@app.post("/find-nearest-neighbors", summary="Find N Nearest Neighbors", response_model=List[NeighborResult])
def find_neighbors_endpoint(
    query_embedding: List[float] = Query(..., description=f"Face embedding to search for ({K_DIMENSIONS} floats). Example: [0.1, 0.2, ..., 1.28]"),
    n_neighbors: int = Query(1, ge=1, description="Number of nearest neighbors to find.")
):
    """
    Finds the N nearest neighbors for a given query embedding.
    The query_embedding should be passed as a JSON array in the request body if using POST,
    or as a comma-separated list in the URL if GET (FastAPI handles Query for POST body too).
    For simplicity this example uses POST and expects the embedding in the body.
    Adjusted to a POST endpoint for easier embedding input.
    """
    if len(query_embedding) != K_DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"Query embedding must have {K_DIMENSIONS} dimensions.")

    query_embedding_c = (ctypes.c_float * K_DIMENSIONS)(*query_embedding)
    
    query_c_record = FaceRecord()
    ctypes.memmove(query_c_record.embedding, query_embedding_c, ctypes.sizeof(query_embedding_c))
    # query_c_record.person_id não é relevante para a busca, pode deixar em branco ou com um placeholder

    # Alocar buffer para os resultados em Python
    results_array_type = FaceRecord * n_neighbors
    results_buffer = results_array_type()

    try:
        num_found = lib.EXPORT_buscar_n_vizinhos(query_c_record, n_neighbors, results_buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed during k-NN search: {str(e)}")

    response_data: List[NeighborResult] = []
    for i in range(num_found):
        record = results_buffer[i]
        response_data.append(
            NeighborResult(
                person_id=record.person_id.decode('utf-8', errors='ignore'),
                embedding=list(record.embedding), # Converte de volta para lista de Python floats
                distance=record.distance_to_query
            )
        )
    
    if not response_data:
        return [] # Ou HTTPException(status_code=404, detail="No neighbors found or tree is empty.")
        
    return response_data

# Adicionar um endpoint para inicializar a árvore explicitamente ao iniciar o servidor (opcional)
@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup: Initializing KD-Tree...")
    lib.EXPORT_kdtree_construir()
    print("KD-Tree initialized via FastAPI startup event.")

# Para rodar: uvicorn app:app --reload