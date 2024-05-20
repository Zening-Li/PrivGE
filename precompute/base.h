#ifndef BASE_H_
#define BASE_H_

#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "utils.h"

using namespace std;
namespace py = pybind11;


class Graph{
public:
    vector<vector<int>> adj;
    vector<int> degree;
    float* feat_ptr;
    int num_nodes;
    long dimension;
    int num_threads;
    float alpha;
    float rmax;
    float rrr;
    vector<int> random_start_node;
    vector<vector<float>> negative_feature;
    vector<float> positive_row_sum;
    vector<float> negative_row_sum;
    uint32_t seed = time(0);

    Graph(py::array_t<int> edges, py::array_t<float> features,
        int nodes, int num_workers, float decay, float error, float rrz){
        cout << "c++..." << endl;
        py::buffer_info edges_buf = edges.request();
        int* edges_ptr = (int*)edges_buf.ptr;
        num_nodes = nodes;
        num_threads = num_workers;
        alpha = decay;
        rmax = error;
        rrr = rrz;
        // load adj
        adj = vector<vector<int>>(num_nodes);
        degree = vector<int>(num_nodes, 0);
        long num_edges = edges_buf.shape[1];
        for(int i=0; i<num_edges; i++){
            adj[edges_ptr[num_edges+i]].push_back(edges_ptr[i]);
            degree[edges_ptr[i]]++;
        }
        // cout << "load adj success." << endl;
        // load attr
        py::buffer_info feat_buf = features.request();
        feat_ptr = (float*)feat_buf.ptr;
        // num_nodes x dims
        int num_rows = feat_buf.shape[0];
        int num_cols = feat_buf.shape[1];
        // cout << "num_rows: " << num_rows << " num_cols: " << num_cols << endl;
        dimension = num_cols;
        if (num_threads > dimension){
            num_threads = dimension;
        }
        random_start_node = vector<int>(dimension);
        positive_row_sum = vector<float>(dimension, 0.);
        negative_row_sum = vector<float>(dimension, 0.);
        for (int i = 0; i < dimension; i++){
            random_start_node[i] = i;
        }
        random_shuffle(random_start_node.begin(), random_start_node.end());
        negative_feature = vector<vector<float>>(num_cols, vector<float>(num_rows));
        for (int row = 0; row < num_rows; row++){
            for (int col = 0; col < num_cols; col++){
                auto val = feat_ptr[row * dimension + col];
                if (degree[row] > 0){
                    val = val / pow(degree[row], rrr);
                }
                if (val > 0){
                    feat_ptr[row * dimension + col] = val;
                    positive_row_sum[col] += val;
                }
                else{
                    feat_ptr[row * dimension + col] = 0;
                    negative_feature[col][row] = -val;
                    negative_row_sum[col] += -val;
                }
            }
        }
        for (int i = 0; i < num_cols; i++){
            if (positive_row_sum[i] == 0){
                positive_row_sum[i] = 1;
            }
            if (negative_row_sum[i] == 0){
                negative_row_sum[i] = 1;
            }
        }
        // cout << "load attr success." <<endl;
    }
    // generate the random number from [0,1]
    // RAND_MAX: The largest number rand will return (same as INT_MAX)
    float rand_0_1(){
        return rand_r(&seed) % RAND_MAX / (float)RAND_MAX;
    }
    int rand_max(){
        return rand_r(&seed);
    }
};


class Base : public Graph
{
public:
    Base(py::array_t<int> edges, py::array_t<float> features, 
        int nodes, int num_workers, float decay, float error, float rrz) : Graph(edges, features, nodes, num_workers, decay, error, rrz){

    }

    void BackwardPush(int start, int end){
        float *reserve = new float[num_nodes];
        Node_Set *candidate_set1 = new Node_Set(num_nodes);
        Node_Set *candidate_set2 = new Node_Set(num_nodes);
        for (int i = start; i < end; i++){
            int w = random_start_node[i];
            float row_sum_pos = positive_row_sum[w];
            float row_sum_neg = negative_row_sum[w];
            for (int j = 0; j < num_nodes; j++){
                if (feat_ptr[j*dimension+w] > rmax * row_sum_pos){
                    candidate_set1->Push(j);
                }
                if (negative_feature[w][j] > rmax * row_sum_neg){
                    candidate_set2->Push(j);
                }
                reserve[j] = 0;
            }
            while (candidate_set1->KeyNumber != 0){
                int old_node = candidate_set1->Pop();
                float old_residue = feat_ptr[old_node*dimension+w];
                float rpush = (1 - alpha) * old_residue;
                reserve[old_node] += alpha * old_residue;
                feat_ptr[old_node*dimension+w] = 0;
                for (auto new_node : adj[old_node]){
                    feat_ptr[new_node*dimension+w] += rpush / degree[new_node];
                    if (feat_ptr[new_node*dimension+w] > rmax * row_sum_pos){
                        candidate_set1->Push(new_node);
                    }
                }
            }
            while (candidate_set2->KeyNumber != 0){
                int old_node = candidate_set2->Pop();
                float old_residue = negative_feature[w][old_node];
                float rpush = (1 - alpha) * old_residue;
                reserve[old_node] -= alpha * old_residue;
                negative_feature[w][old_node] = 0;
                for (auto new_node : adj[old_node]){
                    negative_feature[w][new_node] += rpush / degree[new_node];
                    if (negative_feature[w][new_node] > rmax * row_sum_neg){
                        candidate_set2->Push(new_node);
                    }
                }
            }
            for (long k = 0; k < num_nodes; k++){
                float t = alpha*(feat_ptr[k*dimension+w]-negative_feature[w][k])+reserve[k];
                if (degree[k] > 0){
                    t *= pow(degree[k], rrr);
                }
                feat_ptr[k*dimension+w] = t;
            }
            vector<float>().swap(negative_feature[w]);
            candidate_set1->Clean();
            candidate_set2->Clean();
        }
        delete[] reserve;
    }

    void PagerankPush(){
        struct timeval time_start, time_end;
        double time_cost;
        gettimeofday(&time_start, NULL);
        vector<thread> threads;
        int i;
        int start;
        int end = 0;
        for (i = 1; i <= dimension % num_threads; i++){
            start = end;
            end += ceil((double)dimension / num_threads);
            threads.push_back(thread(&Base::BackwardPush, this, start, end));
        }
        for (; i <= num_threads; i++){
            start = end;
            end += dimension / num_threads;
            threads.push_back(thread(&Base::BackwardPush, this, start, end));
        }
        for (int t = 0; t < num_threads; t++){
            threads[t].join();
        }
        vector<vector<int>>().swap(adj);
        vector<int>().swap(degree);
        vector<vector<float>>().swap(negative_feature);
        vector<thread>().swap(threads);
        gettimeofday(&time_end, NULL);
        time_cost = time_end.tv_sec - time_start.tv_sec + (time_end.tv_usec - time_start.tv_usec) / 1000000.0;
        cout << "pre-computation cost: " << time_cost << " s" << endl;
    }
};

#endif