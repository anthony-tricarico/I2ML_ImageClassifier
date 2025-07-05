# Convert to numpy arrays
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def create_dict_submission(gallery_embeddings, query_embeddings, similarity_matrix, query_paths, gallery_paths, k: int=5):
    submission = dict()
    
    gallery_embeddings = np.vstack(gallery_embeddings)
    query_embeddings = np.vstack(query_embeddings)

    top_k = k  # You can change this to any value (e.g., 1, 3, 10)

    # Compute top-k most similar gallery indices for each query
    topk_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]

    # Display results
    for i, indices in enumerate(topk_indices):
        print(f"\nQuery image: {query_paths[i]}")
        print("Top {} retrieved gallery images:".format(top_k))
        # finds index of the rightmost slash so that strings can be subset to only include the name
        # of the file and not the entire path leading to it
        idx_last_slash = str(query_paths[i]).rfind("/")
        submission[str(query_paths[i][idx_last_slash+1:])] = list()
        for rank, gallery_idx in enumerate(indices):
            print(f"  Rank {rank+1}: {gallery_paths[gallery_idx]}")
            # finds index of the rightmost slash so that strings can be subset to only include the name
            # of the file and not the entire path leading to it
            idx_last_slash_res = str(gallery_paths[gallery_idx]).rfind("/")
            submission[str(query_paths[i][idx_last_slash+1:])].append(gallery_paths[gallery_idx][idx_last_slash_res+1:])

    return submission