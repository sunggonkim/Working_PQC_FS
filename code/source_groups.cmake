# Aggregate production source manifests. Concrete file paths live beside the
# owning domain so module moves do not require editing this central file.
include(${CODE_DIR}/frontend/sources.cmake)
include(${CODE_DIR}/crypto/sources.cmake)
include(${CODE_DIR}/storage/sources.cmake)
include(${CODE_DIR}/runtime/sources.cmake)
include(${CODE_DIR}/fs/sources.cmake)
include(${CODE_DIR}/support/sources.cmake)
