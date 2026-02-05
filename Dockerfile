# Start from the specified base image
FROM docker.all-hands.dev/all-hands-ai/runtime:0.55-nikolaik


# Install necessary libraries
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    jupyter \
    notebook \
    tensorflow \
    torch \
    torchvision \
    torchaudio \
    xgboost \
    lightgbm \
    dataset