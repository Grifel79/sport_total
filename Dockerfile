FROM debian:latest

LABEL description="Build container - sport_total_app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    file \
    gcc \
    g++ \
    git \
    tar \
    unzip \
    wget \
    --fix-missing \
    && rm -rf /var/lib/apt/lists/*

RUN cd usr/local \
    && wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip

RUN cd usr/local \
    && wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.3.zip \
    && unzip opencv.zip \
    && mv opencv-4.5.3 opencv \
    && mkdir -p build && cd build \
    && cmake ../opencv \
    -D BUILD_DOCS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_SHARED_LIBS=OFF \ 
    -D BUILD_TESTS=OFF \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D FORCE_VTK=OFF \
    -D WITH_FFMPEG=OFF \
    -D WITH_GDAL=OFF \ 
    -D WITH_IPP=OFF \
    -D WITH_OPENEXR=OFF \
    -D WITH_OPENGL=OFF \ 
    -D WITH_QT=OFF \
    -D WITH_TBB=OFF \ 
    -D WITH_XINE=OFF \ 
    -D BUILD_JPEG=ON  \
    -D BUILD_TIFF=ON \
    -D BUILD_PNG=ON \
    && make -j4\
    && make install

COPY ./src /src
WORKDIR /src
RUN mkdir out \
    && cd out \
    && cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch .. \
    && make

WORKDIR /src/out

CMD ./sport_total_app ../snake.jpeg

