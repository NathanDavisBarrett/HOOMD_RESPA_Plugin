// Include the defined classes that are to be exported to python
#include "RespaIntegrator.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

// specify the python module. Note that the name must explicitly match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
PYBIND11_MODULE(_respa_plugin, m)
{
    export_RespaIntegrator(m);

#ifdef ENABLE_HIP
    export_RespaGPU(m);
#endif
}
