# common logic accross linux and darwin

init_vars() {
    case "${GOARCH}" in
    "amd64")
        ARCH="x86_64"
        ;;
    "arm64")
        ARCH="arm64"
        ;;
    *)
        ARCH=$(uname -m | sed -e "s/aarch64/arm64/g")
    esac

    LLAMACPP_DIR=../llama.cpp
    CMAKE_DEFS=""
    CMAKE_TARGETS="--target ext_server"
    if echo "${CGO_CFLAGS}" | grep -- '-g' >/dev/null; then
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_VERBOSE_MAKEFILE=on -DLLAMA_GPROF=on -DLLAMA_SERVER_VERBOSE=on ${CMAKE_DEFS}"
    else
        # TODO - add additional optimization flags...
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_SERVER_VERBOSE=off ${CMAKE_DEFS}"
    fi
    case $(uname -s) in 
    "Darwin")
        LIB_EXT="dylib"
        WHOLE_ARCHIVE="-Wl,-force_load"
        NO_WHOLE_ARCHIVE=""
        GCC_ARCH="-arch ${ARCH}"
        ;;
    "Linux")
        LIB_EXT="so"
        WHOLE_ARCHIVE="-Wl,--whole-archive"
        NO_WHOLE_ARCHIVE="-Wl,--no-whole-archive"

        # Cross compiling not supported on linux - Use docker
        GCC_ARCH=""
        ;;
    *)
        ;;
    esac
    if [ -f /etc/nv_tegra_release ] ; then
        TEGRA_DEVICE=1
        echo "Tegra device detected: ${JETSON_MODEL}, L4T Version ${JETSON_L4T}, Jetpack Version ${JETSON_JETPACK}"
    fi
    if [ -z "${CMAKE_CUDA_ARCHITECTURES}" ] ; then
        if [ -n ${TEGRA_DEVICE} ]; then
            echo "CMAKE_CUDA_ARCHITECTURES unset, values are needed for Tegra devices. Using default values defined at https://github.com/dusty-nv/jetson-containers/blob/master/jetson_containers/l4t_version.py"
        fi
        # Tegra devices will fail generate unless architectures are set:
        # Nano/TX1 = 5.3, TX2 = 6.2, Xavier = 7.2, Orin = 8.7
        # L4T_VERSION.major >= 36:    # JetPack 6
        #     CUDA_ARCHITECTURES = [87]
        # L4T_VERSION.major >= 34:  # JetPack 5
        #     CUDA_ARCHITECTURES = [72, 87]
        # L4T_VERSION.major == 32:  # JetPack 4
        #     CUDA_ARCHITECTURES = [53, 62, 72]
        case $(echo "${JETSON_JETPACK}" | cut -d"." -f1) in
        "6")
            echo "Jetpack 6 detected. Setting CMAKE_CUDA_ARCHITECTURES='87'"
            CMAKE_CUDA_ARCHITECTURES="87"
            ;;
        "5")
            echo "Jetpack 5 detected. Setting CMAKE_CUDA_ARCHITECTURES='72;87'"
            CMAKE_CUDA_ARCHITECTURES="72;87"
            ;;
        "4")
            echo "Jetpack 4 detected. Setting CMAKE_CUDA_ARCHITECTURES='53;62;72'"
            CMAKE_CUDA_ARCHITECTURES="53;62;72"
            ;;
        *)
            CMAKE_CUDA_ARCHITECTURES="50;52;61;70;75;80"
            ;;
        esac
    fi
}

git_module_setup() {
    if [ -n "${OLLAMA_SKIP_PATCHING}" ]; then
        echo "Skipping submodule initialization"
        return
    fi
    # Make sure the tree is clean after the directory moves
    if [ -d "${LLAMACPP_DIR}/gguf" ]; then
        echo "Cleaning up old submodule"
        rm -rf ${LLAMACPP_DIR}
    fi
    git submodule init
    git submodule update --force ${LLAMACPP_DIR}

}

apply_patches() {
    # Wire up our CMakefile
    if ! grep ollama ${LLAMACPP_DIR}/examples/server/CMakeLists.txt; then
        echo 'include (../../../ext_server/CMakeLists.txt) # ollama' >>${LLAMACPP_DIR}/examples/server/CMakeLists.txt
    fi

    if [ -n "$(ls -A ../patches/*.diff)" ]; then
        # apply temporary patches until fix is upstream
        for patch in ../patches/*.diff; do
            for file in $(grep "^+++ " ${patch} | cut -f2 -d' ' | cut -f2- -d/); do
                (cd ${LLAMACPP_DIR}; git checkout ${file})
            done
        done
        for patch in ../patches/*.diff; do
            (cd ${LLAMACPP_DIR} && git apply ${patch})
        done
    fi

    # Avoid duplicate main symbols when we link into the cgo binary
    sed -e 's/int main(/int __main(/g' <${LLAMACPP_DIR}/examples/server/server.cpp >${LLAMACPP_DIR}/examples/server/server.cpp.tmp &&
        mv ${LLAMACPP_DIR}/examples/server/server.cpp.tmp ${LLAMACPP_DIR}/examples/server/server.cpp
}

build() {
    cmake -S ${LLAMACPP_DIR} -B ${BUILD_DIR} ${CMAKE_DEFS}
    cmake --build ${BUILD_DIR} ${CMAKE_TARGETS} -j8
    mkdir -p ${BUILD_DIR}/lib/
    g++ -fPIC -g -shared -o ${BUILD_DIR}/lib/libext_server.${LIB_EXT} \
        ${GCC_ARCH} \
        ${WHOLE_ARCHIVE} ${BUILD_DIR}/examples/server/libext_server.a ${NO_WHOLE_ARCHIVE} \
        ${BUILD_DIR}/common/libcommon.a \
        ${BUILD_DIR}/libllama.a \
        -Wl,-rpath,\$ORIGIN \
        -lpthread -ldl -lm \
        ${EXTRA_LIBS}
}

compress_libs() {
    echo "Compressing payloads to reduce overall binary size..."
    pids=""
    rm -rf ${BUILD_DIR}/lib/*.${LIB_EXT}*.gz
    for lib in ${BUILD_DIR}/lib/*.${LIB_EXT}* ; do
        gzip --best -f ${lib} &
        pids+=" $!"
    done
    echo 
    for pid in ${pids}; do
        wait $pid
    done
    echo "Finished compression"
}

# Keep the local tree clean after we're done with the build
cleanup() {
    (cd ${LLAMACPP_DIR}/examples/server/ && git checkout CMakeLists.txt server.cpp)

    if [ -n "$(ls -A ../patches/*.diff)" ]; then
        for patch in ../patches/*.diff; do
            for file in $(grep "^+++ " ${patch} | cut -f2 -d' ' | cut -f2- -d/); do
                (cd ${LLAMACPP_DIR}; git checkout ${file})
            done
        done
    fi
}
