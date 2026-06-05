import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/parkinsons-drawings")

print("Path to dataset files:", path)