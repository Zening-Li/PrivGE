#include "push.h"


py::array_t<float> calc_pagerank_push(py::array_t<int> edges, 
    py::array_t<float> features, int nodes, float alpha, float rmax, float rrr)
{
    int num_threads = 32;
    Base graph(edges, features, nodes, num_threads, alpha, rmax, rrr);
    graph.PagerankPush();
    int num_rows = graph.num_nodes;
    int num_cols = int(graph.dimension);
    py::buffer_info bufinfo(
        // pointer
        graph.feat_ptr,
        // size of underlying scalar type
        sizeof(float),
        // python struct-style format descriptor
        py::format_descriptor<float>::format(),
        // number of dimensions
        2,
        // buffer dimensions
        {num_rows, num_cols},
        // strides (in bytes) for each index
        {sizeof(float) * num_cols, sizeof(float)}
    );
    return py::array_t<float>(bufinfo);
}


PYBIND11_MODULE(push, m){
    m.doc() = "calculate pagerank via push method";
    m.def("calc_pagerank_push", &calc_pagerank_push);
}
