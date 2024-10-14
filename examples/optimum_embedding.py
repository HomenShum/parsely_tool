from llama_index.embeddings.huggingface_optimum import OptimumEmbedding

if not os.path.exists("./bge_m3_onnx"):
    OptimumEmbedding.create_and_save_optimum_model(
        # "BAAI/bge-small-en-v1.5", "./bge_onnx"
        "BAAI/bge-m3", "./bge_m3_onnx"
    )

embed_model = OptimumEmbedding(folder_name="./bge_m3_onnx")
embeddings = embed_model.get_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])
