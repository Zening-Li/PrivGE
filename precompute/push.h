#ifndef PUSH_H_
#define PUSH_H_

#include "base.h"


py::array_t<float> calc_pagerank_push(py::array_t<int> edges,
    py::array_t<float> features, int nodes, float alpha, float rmax, float rrr);

#endif
