add_executable(test-model-forward model-forward.cpp)
target_include_directories(test-model-forward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor)
target_link_libraries(test-model-forward PRIVATE fusealign_model fusealign_host)
