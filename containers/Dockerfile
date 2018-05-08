FROM ubuntu:16.04 
RUN apt-get update && apt-get install -y build-essential \
    autotools-dev \ 
    autoconf \
    automake \
    vim \
    git \
    python \
    curl \
    zip \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && git clone https://github.com/spack/spack.git \
    && git clone https://github.com/LLNL/lbann.git \
    && cd lbann \
    && mkdir spack_build \
    && cd spack_build \
    && /spack/bin/spack  -k install --dirty --no-checksum gcc@7.1.0 \
    && GCC="$(/spack/bin/spack location --install-dir gcc@7.1.0)" \
    && /spack/bin/spack compiler add $GCC \
    && /spack/bin/spack -k setup lbann@local %gcc@7.1.0 build_type=Release ^elemental@hydrogen-develop ^openmpi@2.0.2 ^cmake@3.9.0 \ 
    && mkdir docker_build \ 
    && cd docker_build \
    && ../spconfig.py ../.. \
    && make -j3 all \
    && /spack/bin/spack uninstall -y lbann
    
