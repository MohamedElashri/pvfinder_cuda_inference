# PVFinder CUDA Inference

CUDA implementation of PVFinder neural network inference for vertex finding.

## Project Structure
```
pvfinder_cuda/
├── CMakeLists.txt        # Build configuration
├── include/
│   ├── common.hpp        # Common utilities, tensor class, CUDA helpers
│   ├── model.hpp         # Neural network model definition
│   ├── inference.hpp     # Inference engine interface
│   └── layers/
│       ├── activation.hpp # Activation functions (LeakyReLU, Softplus)
│       ├── conv.hpp      # Convolution layer
│       ├── deconv.hpp    # Deconvolution (transpose conv) layer
│       ├── linear.hpp    # Linear (fully connected) layer
│       └── pooling.hpp   # Pooling operations
└── src/
    ├── inference.cu      # Main inference implementation
    ├── model.cu         # Model implementation
    └── layers/
        ├── activation.cu # Activation functions implementation
        ├── conv.cu       # Convolution implementation
        ├── deconv.cu     # Deconvolution implementation
        ├── linear.cu     # Linear layer implementation
        └── pooling.cu    # Pooling implementation
```

## Dependencies
- CUDA Toolkit (tested with 12.1)
- cuDNN
- cnpy (for loading numpy files)
- zlib (required by cnpy)

## Setup Dependencies

Save this script as `setup_dependencies.sh`:
```bash
#!/bin/bash
# Set up variables
INSTALL_DIR="$HOME/local"
ZLIB_VERSION="1.3.1"
CNPY_REPO="https://github.com/rogersce/cnpy.git"

# Create the local directory if it doesn't exist
mkdir -p $INSTALL_DIR

# Step 1: Install zlib in ~/local
cd $HOME
if [ ! -d "zlib-$ZLIB_VERSION" ]; then
    echo "Downloading and installing zlib version $ZLIB_VERSION..."
    wget http://www.zlib.net/zlib-$ZLIB_VERSION.tar.gz
    tar -xvzf zlib-$ZLIB_VERSION.tar.gz
    cd zlib-$ZLIB_VERSION
    ./configure --prefix=$INSTALL_DIR
    make
    make install
else
    echo "zlib-$ZLIB_VERSION already downloaded and installed."
fi

# Step 2: Clone and build cnpy
cd $HOME
if [ ! -d "cnpy" ]; then
    echo "Cloning cnpy repository..."
    git clone $CNPY_REPO
else
    echo "cnpy repository already cloned."
fi
cd cnpy

# Step 3: Configure and build cnpy with custom zlib and system toolchain
echo "Configuring and building cnpy..."
cmake . \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DZLIB_LIBRARY=$INSTALL_DIR/lib/libz.so \
    -DZLIB_INCLUDE_DIR=$INSTALL_DIR/include \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_INSTALL_LIBDIR=$INSTALL_DIR/lib \
    -DCMAKE_INSTALL_BINDIR=$INSTALL_DIR/bin \
    -DCMAKE_INSTALL_INCLUDEDIR=$INSTALL_DIR/include
make
make install
```

Run the setup script:
```bash
chmod +x setup_dependencies.sh
./setup_dependencies.sh
```

## Build Instructions
```bash
# Create build directory
mkdir build
cd build

# Configure with correct paths
cmake .. \
    -DCMAKE_INSTALL_PREFIX=~/local \
    -DZLIB_INCLUDE_DIR=~/local/include \
    -DZLIB_LIBRARY=~/local/lib/libz.so \
    -DCNPY_INCLUDE_DIR=~/local/include \
    -DCNPY_LIBRARY=~/local/lib/libcnpy.so

# Build
make
```

## Usage
```bash
# Show help
./pvfinder -h

# Basic usage
./pvfinder -w weights.npz -d data.npy

# With custom output path
./pvfinder -w weights.npz -d data.npy -o results.csv
```

### Command Line Options
- `-w, --weights FILE` : Model weights file (NPZ format)
- `-d, --data FILE`    : Validation data file (NPY format)
- `-o, --out FILE`     : Output file [default: ./output.csv]
- `-h, --help`         : Show help message

### Input Format
- Weights: NPZ file containing model parameters
- Data: NPY file containing validation data (shape: [N, 9, 250])

### Output Format
CSV file with format:
```
EventID,bin0,bin1,...,bin99
0,0.123,0.456,...,0.789
1,0.234,0.567,...,0.890
...
```

## Notes
- Default output path is ./output.csv
- Each row represents one event with 100 KDE values
- Output values represent probability distribution of vertex positions
- Bin values correspond to 2mm steps from -100mm to +100mm

## Example Command
```bash
./pvfinder \
    -w /path/to/model_weights.npz \
    -d /path/to/validation_data.npy \
    -o /path/to/results.csv
```

## Troubleshooting
If you see library-related errors during runtime, make sure to update your LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
```