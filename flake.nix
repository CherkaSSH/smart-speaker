{
  description = "Homemade smart speaker: local STT, LLM and TTS";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      lib = pkgs.lib;

      # LFM3 needs llama.cpp b11960 (v0.8.0) or newer; nixpkgs ships v0.6.0.
      # Built here without the web UI, with Vulkan for the Radeon 780M iGPU.
      llama-cpp = pkgs.stdenv.mkDerivation {
        pname = "llama-cpp-vulkan";
        version = "b12017";
        src = pkgs.fetchFromGitHub {
          owner = "ggml-org";
          repo = "llama.cpp";
          rev = "5047748f114564fcf271e6a648f60e2d2eae6264";
          hash = "sha256-x8uR3FebP7TuRwB51MjR7huXf9fOaflHs1e4d4Jw+9k=";
        };
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config shaderc python3 ];
        buildInputs = with pkgs; [ vulkan-headers vulkan-loader spirv-headers openssl ];
        cmakeFlags = [
          "-DGGML_VULKAN=ON"
          "-DGGML_NATIVE=ON"
          "-DLLAMA_BUILD_UI=OFF"
          "-DLLAMA_USE_PREBUILT_UI=OFF"
          "-DLLAMA_BUILD_TESTS=OFF"
          "-DLLAMA_BUILD_EXAMPLES=OFF"
          "-DLLAMA_BUILD_NUMBER=12017"
        ];
      };

      # NVIDIA's native runtime for Nemotron/Parakeet ASR, with Vulkan so it can
      # run on the Radeon 780M ([asr] device). The CMake build applies its own
      # ggml patch series with git, so git must be on the build PATH.
      nemo-speech = pkgs.stdenv.mkDerivation {
        pname = "nemo-speech";
        version = "0.3.0-20999cc";
        src = pkgs.fetchFromGitHub {
          owner = "NVIDIA";
          repo = "NeMo-Speech.cpp";
          rev = "20999cc921d2b14aa58708f61756a8b2e787ba20";
          fetchSubmodules = true;
          hash = "sha256-UJdFdVYrvSsPntcUSP6oiPFz2/4X/13upncg0B1iepA=";
        };
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config git python3 shaderc ];
        buildInputs = with pkgs; [ sentencepiece abseil-cpp protobuf alsa-lib openssl
                                   vulkan-headers vulkan-loader spirv-headers ];
        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=Release"
          "-DGGML_NATIVE=ON"
          "-DGGML_LLAMAFILE=ON"
          "-DGGML_VULKAN=ON"
          "-DNEMO_SPEECH_GGML_PATCHED=ON"
          "-DNEMO_SPEECH_BUILD_ASR=ON"
          "-DNEMO_SPEECH_BUILD_DIAR=OFF"
          "-DNEMO_SPEECH_BUILD_TTS=OFF"
          "-DNEMO_SPEECH_BUILD_NMT=OFF"
          "-DNEMO_SPEECH_BUILD_SPEECHLM=ON"
          "-DNEMO_SPEECH_BUILD_HTTP=ON"
          "-DNEMO_SPEECH_BUILD_GRPC=OFF"
          "-DNEMO_SPEECH_BUILD_TESTS=OFF"
        ];
      };

      # Neural acoustic echo canceller (LocalVQE, Apache-2.0): removes our own music
      # and speech from the mic while keeping the user's voice. The voice service
      # loads liblocalvqe.so with ctypes; on x86 ggml picks its CPU variant at run
      # time from the libggml-cpu-*.so files installed next to it.
      localvqe = pkgs.stdenv.mkDerivation {
        pname = "localvqe";
        version = "0.1.0-7f61cd0";
        src = pkgs.fetchFromGitHub {
          owner = "localai-org";
          repo = "LocalVQE";
          rev = "7f61cd05095d42e56a135b303f5bb4d761da22b9";
          fetchSubmodules = true;
          hash = "sha256-iHMnP1EMiIjb0aofGJUKjMm8Z11W36SVYNYq0ZYowAc=";
        };
        sourceRoot = "source/ggml";
        # CMake patches a GRU op into the vendored ggml with `git apply`.
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config git autoPatchelfHook ];
        buildInputs = with pkgs; [ libsndfile stdenv.cc.cc.lib ];
        cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" "-DLOCALVQE_BUILD_SHARED=ON"
                       "-DCMAKE_SKIP_BUILD_RPATH=ON" ];  # autoPatchelfHook sets it from $out/lib
        installPhase = ''
          mkdir -p $out/bin $out/lib
          cp -P bin/*.so* $out/lib/
          cp bin/localvqe $out/bin/
        '';
      };

      # Wheels from uv (torch, onnxruntime, ...) dlopen these at run time.
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        portaudio
        alsa-lib
        libsndfile
      ];
    in
    {
      packages.${system} = {
        inherit llama-cpp nemo-speech localvqe;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          alsa-utils
          ffmpeg
          llama-cpp
          nemo-speech
          localvqe
        ];
        LOCALVQE_LIB = "${localvqe}/lib/liblocalvqe.so";
        LD_LIBRARY_PATH = lib.makeLibraryPath runtimeLibs;
        UV_PYTHON = "${pkgs.python312}/bin/python3.12";
        UV_PYTHON_DOWNLOADS = "never";
        # The host has no hardware.graphics; use nixpkgs' RADV for the 780M.
        VK_ICD_FILENAMES = "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json";
        # Standard ALSA config plus the speaker's shared (dmix/dsnoop) devices.
        ALSA_CONFIG_PATH = "${pkgs.alsa-lib}/share/alsa/alsa.conf:${./alsa/speaker.conf}";
      };
    };
}
