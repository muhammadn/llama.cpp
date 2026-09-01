set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml cpp-httplib)
# for common/http.h only -- deliberately not linking the rest of libcommon,
# to keep this binary a minimal, standalone RPC server
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/common ${CMAKE_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
