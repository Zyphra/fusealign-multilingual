add_executable(test-host host-tests.cpp)
target_link_libraries(test-host PRIVATE fusealign_host)
add_test(NAME test-host COMMAND test-host)

# JSON-lines driver for host_parity.py (not a ctest: it needs the Python reference package).
add_executable(test-host-parity host-parity.cpp)
target_include_directories(test-host-parity PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(test-host-parity PRIVATE fusealign_host)
