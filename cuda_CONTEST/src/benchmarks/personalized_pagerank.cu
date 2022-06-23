// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <sstream>
#include "personalized_pagerank.cuh"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

//////////////////////////////
//////////////////////////////

// Write GPU kernel here!
int *x_gpu;
int *y_gpu;
float *val_gpu;
int *V_gpu;
int *E_gpu;
float *pr_gpu;
float *pr_tmp;
int *dangling_gpu;

int *y_csr;
int *x_ptr;
float *val_csr;

float *val_float;
float *pr_float;

//#define log_warp_size 5;
//#define warp_size 32;
void spmv_sim(
                    const int *x, //x_ptr
                    const int *y, 
                    const float *val, 
                    const float *vec, 
                    float *result,
                    int V,
                    int pers,
                    float alpha,
                    float beta,
                    int id) {
	float res = 0;
    if (id < V) {
        
        int els = x[id+1] - x[id];
        for(int i = 0; i < els; i++) {
            res += val[x[id] + i] * vec[y[x[id] + i]];
            //if (vec[y[x[id] - els +i]] != 0)
                //std::cout << "vec[y[x[id] - els +i]] = "<< vec[y[x[id] - els +i]] << std::endl;
            if (i == els -1){
                res = (res * alpha) + beta;
                //check personalization vector
                if (id == pers) {
                    res = res + (1 - alpha);
                }
            }
        }
        //syncthreads before write?
        
        if (res == 1.0/V){
            std::cout << "no change for pr_" << id << std::endl;
        }

        std::cout << "new pr of v_" << id << " : " << res << std::endl;
        result[id] = res;

    }
}

__global__ void spmv(
                    const int *x, //x_ptr
                    const int *y, 
                    const float *val, 
                    const float *vec, 
                    float *result,
                    int V,
                    int pers,
                    float alpha,
                    float beta) {
	int id = threadIdx.x+blockIdx.x*blockDim.x;
	float res = 0;
    if (id < V - 1) {
        int els = x[id] - (id == 0 ? 0 : x[id-1]);

        for(int i = 0; i < els; i++) {
            res += val[x[id] + i] * vec[y[x[id]+i]];

            if (x[i] != x[i + 1])
                res = (res * alpha) + beta;
            //check personalization vector
            if (x[id] == pers)
                res += 1 - alpha;
        }
        //syncthreads before write?
        result[id] = res;
    }
}

__global__ void dangle_factor(
                            const int *x, //x_ptr
                            const int *y, 
                            const float *pr,
                            int V,
                            float result) {

    extern __shared__ float buffer[];
	//assert(sizeof(buffer) >= ((V >> log_warp_size) + 1));
    int id = threadIdx.x+blockIdx.x*blockDim.x;
    int partial = 0;
    int warp_size = 32;
    int log_warp_size = 5;

    if (id < V - 1) {
        int els = x[id] - (id == 0 ? 0 : x[id-1]);
        if (els == 0) {
            partial = pr[y[x[id]]];
        }
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
        
        if ((id % warp_size) == 0) {
            buffer[(id >> log_warp_size)] = partial;
        }

        els = (V << log_warp_size) + (V % warp_size);

        while (els > 1) {
            //if (id > els)
            //    return;
            partial = buffer[id];

            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }

            if (id % (warp_size) == 0)
                buffer[id >> log_warp_size] = partial;
        
            els = (els << log_warp_size) + (els % warp_size);
        }
        result = buffer[0];
    }
}

//////////////////////////////
//////////////////////////////

// CPU Utility functions;

inline float dot_product_cpu_float(const int *a, const float *b, const int N) {
    float result = 0;
    for (int i = 0; i < N; i++) {
        result += a[i] * b[i];
    }
    return result;
}

inline void axpb_personalized_cpu_float(
    float alpha, float *x, float beta,
    const int personalization_vertex, float *result, const int N) {
    float one_minus_alpha = 1 - alpha;
    for (int i = 0; i < N; i++) {
        result[i] = alpha * x[i] + beta + ((personalization_vertex == i) ? one_minus_alpha : 0.0);
    }
}

inline float euclidean_distance_cpu_float(const float *x, const float *y, const int N) {
    double result = 0;
    for (int i = 0; i < N; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return std::sqrt(result);
}

bool sort_csr(const std::tuple<int, int, float>& a, 
              const std::tuple<int, int, float>& b)
{
    if (std::get<1>(a) != std::get<1>(b))
        return (std::get<1>(a) < std::get<1>(b));
    else
        return std::get<0>(a) < std::get<0>(b);
}

// Read the input graph and initialize it;
void PersonalizedPageRank::initialize_graph() {
    // Read the graph from an MTX file;
    int num_rows = 0;
    int num_columns = 0;
    read_mtx(graph_file_path.c_str(), &x, &y, &val,
        &num_rows, &num_columns, &E, // Store the number of vertices (row and columns must be the same value), and edges;
        true,                        // If true, read edges TRANSPOSED, i.e. edge (2, 3) is loaded as (3, 2). We set this true as it simplifies the PPR computation;
        false,                       // If true, read the third column of the matrix file. If false, set all values to 1 (this is what you want when reading a graph topology);
        debug,                 
        false,                       // MTX files use indices starting from 1. If for whatever reason your MTX files uses indices that start from 0, set zero_indexed_file=true;
        true                         // If true, sort the edges in (x, y) order. If you have a sorted MTX file, turn this to false to make loading faster;
    );
    if (num_rows != num_columns) {
        if (debug) std::cout << "error, the matrix is not squared, rows=" << num_rows << ", columns=" << num_columns << std::endl;
        exit(-1);
    } else {
        V = num_rows;
    }
    if (debug) std::cout << "loaded graph, |V|=" << V << ", |E|=" << E << std::endl;

    // Compute the dangling vector. A vertex is not dangling if it has at least 1 outgoing edge;
    dangling.resize(V);
    std::fill(dangling.begin(), dangling.end(), 1);  // Initially assume all vertices to be dangling;
    for (int i = 0; i < E; i++) {
        // Ignore self-loops, a vertex is still dangling if it has only self-loops;
        if (x[i] != y[i]) dangling[y[i]] = 0;
    }
    // Initialize the CPU PageRank vector;
    pr.resize(V);
    pr_golden.resize(V);
    // Initialize the value vector of the graph (1 / outdegree of each vertex).
    // Count how many edges start in each vertex (here, the source vertex is y as the matrix is transposed);
    int *outdegree = (int *) calloc(V, sizeof(int));
    for (int i = 0; i < E; i++) {
        outdegree[y[i]]++;
    }
    // Divide each edge value by the outdegree of the source vertex;
    for (int i = 0; i < E; i++) {
        val[i] = 1.0 / outdegree[y[i]];  
    }
    free(outdegree);
}

//////////////////////////////
//////////////////////////////

// Allocate data on the CPU and GPU;
void PersonalizedPageRank::alloc() {
    // Load the input graph and preprocess it;
    initialize_graph();

    // Allocate any GPU data here;
    // TODO!

    cudaMalloc(&x_gpu, sizeof(int)*(V+1));
    cudaMalloc(&y_gpu, sizeof(int)*E);
    cudaMalloc(&val_gpu, sizeof(float)*E);
    cudaMalloc(&dangling_gpu, sizeof(int)*dangling.size());
    cudaMalloc(&pr_gpu, sizeof(float) * V);
    cudaMalloc(&pr_tmp, sizeof(float) * V);
}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;
    // TODO!
    
    //convert to CSR
    x_ptr = (int *) malloc(sizeof(int)*(V+1));
    memset(x_ptr, 0, sizeof(int) * V);

    for(int i = 0; i < E; i++) {
        x_ptr[x[i]+1]++;
    }

    for(int i = 0; i < V; i++) {
        x_ptr[i+1] += x_ptr[i];
    }

    /*
    int els = 41;
    std::cout << "x_csr: ";
    for(int i = 0; i < els; i++) {
        std::cout << x_csr[i] << ", ";
    }
    std::cout << std::endl << "y_ptr: ";
    for(int i = 0; i < els; i++) {
        if (i < V)
            std::cout << y_ptr[i] << ", ";
    }
    std::cout << std::endl << "val: ";
    for(int i = 0; i < els; i++) {
        std::cout << val_csr[i] << ", ";
    }
    */
}


// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset() {
    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr.begin(), pr.end(), 1.0 / V); 
    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V; 
    if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;

    val_float = (float *) malloc(sizeof(float)*E);
    std::transform(val.begin(), val.end(), val_float, [](double d) -> float {return float(d);});

    float *pr_float = (float *) malloc(sizeof(float)*V);

    float smallest = std::max(float(1.0/V) , float(1.0e-30));
    if (debug)
        std::cout << "initializing pr to: " << smallest << std::endl;
    for (int i = 0; i < V; i++) {
        pr_float[i] = smallest;
    }
    // Do any GPU reset here, and also transfer data to the GPU;
    // TODO!
    cudaMemcpy(x_gpu, x_ptr, sizeof(int) * (V+1), cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y.data(), sizeof(int) * E, cudaMemcpyHostToDevice);
    cudaMemcpy(val_gpu, val_csr, sizeof(float) * E, cudaMemcpyHostToDevice);
    cudaMemcpy(dangling_gpu, dangling.data(), sizeof(int)*dangling.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(pr_gpu, pr_float, sizeof(float) * V, cudaMemcpyHostToDevice);


    // Print out the vector
    std::cout << "E = " << E << " V = " << V << std::endl;

    if (debug) std::cout << "reset successful" << std::endl;
}

void PersonalizedPageRank::execute(int iter) {
    if (debug) std::cout << "starting execution" << std::endl;
    // Do the GPU computation here, and also transfer results to the CPU;
    //TODO! (and save the GPU PPR values into the "pr" array)
    int block_size = 128;
    int n_blocks = max(1, (int) std::ceil(V / block_size));
    //int sharedmemSize = sizeof(float) * ((V >> 5) + 1);
    
    float alpha_f = (float) alpha;
    double* pr_cpu = (double*) malloc(sizeof(double)*V);
    float* pr_tmp_gpu = (float*) malloc(sizeof(float)*V);
    float smallest = std::max(float(1.0/V) , float(1.0e-30));
    for (int i = 0; i < V; i++) {
        pr_cpu[i] = smallest;
    }

    double* pr_tmp_cpu = (double*) malloc(sizeof(double)*V);

    //spvm_sim////////////////////////////////
    float* pr_tmp_sim = (float*) malloc(sizeof(float)*V);
    float* pr_sim = (float*) malloc(sizeof(float)*V);
    for (int i = 0; i < V+1; i++) {
        pr_sim[i] = smallest;
    }
    //////////////////////////////////////////

    bool converged = false;
    while (iter < max_iterations) {    

        
        memset(pr_tmp_cpu, 0, sizeof(double) * V);
        
        spmv_coo_cpu(x.data(), y.data(), val.data(), pr_cpu, pr_tmp_cpu, E);
        double dangling_factor = dot_product_cpu(dangling.data(), pr_cpu, V);
        std::cout << "dang_cpu: "<< dangling_factor << std::endl;
        axpb_personalized_cpu(alpha, pr_tmp_cpu, alpha * dangling_factor / V, personalization_vertex, pr_tmp_cpu, V);
        
        memcpy(pr_cpu, pr_tmp_cpu, sizeof(double) * V);
        
        //if (debug) std::cout << "launching gpu kernel" << std::endl;
        cudaMemset(pr_tmp, 0, sizeof(float) * V);
        
        float dang_gpu = 0;
        /*
        dangle_factor<<<n_blocks, block_size, sharedmemSize>>>(x_gpu, y_gpu, pr_gpu, V, dang_gpu);
        std::cout << "dang_gpu: "<< dang_gpu << std::endl;
        printf("error after dang : %s\n", cudaGetErrorName(cudaGetLastError()));
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        */
        dang_gpu = (float) dangling_factor;
        /*
        spmv<<<n_blocks, block_size>>>(x_gpu, y_gpu, val_gpu, pr_gpu, pr_tmp, V, personalization_vertex,
                                    (float) alpha, (float) alpha * dang_gpu / V);
        printf("error after spmv : %s\n", cudaGetErrorName(cudaGetLastError()));
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        
        //if (debug) std::cout << "exiting gpu kernel" << std::endl;
        
        pr_tmp_gpu = (float *) malloc(sizeof(float)*V);
        cudaMemcpy(pr_tmp_gpu, pr_tmp, sizeof(float) * V, cudaMemcpyDeviceToHost);
        */
        std::cout << "start simulation"<< std::endl;
        memset(pr_tmp_sim, 0, sizeof(float) * V);
        int i = 0;
        for(i = 0; i < V; i++) {
            //std::cout << "pr of v_" << i << " = " << pr_sim[i] << std::endl;
            spmv_sim(x_ptr, y.data(), val_float, pr_sim, pr_tmp_sim, V, personalization_vertex,
                                    (float) alpha, (float) alpha * dang_gpu / V, i);
        }       
        memcpy(pr_sim, pr_tmp_sim, sizeof(float) * V);
/*
        for (int i = 0; i < 20; i++) {
            if (iter < 2) {
            std::cout << "expected = " << pr_tmp_cpu[i] << std::endl;
            std::cout << "found    = " << pr_tmp_sim[i] << std::endl;
            }
        }*/
        

        cudaMemcpy(pr_gpu, pr_tmp, sizeof(float) * V, cudaMemcpyDeviceToDevice);
        
        iter++;
        if (debug) std::cout << "end iter: " << iter << std::endl;
    }
    memcpy(pr.data(), pr_sim, sizeof(double) * V);
    //cudaMemcpy(pr.data(), pr_gpu, sizeof(float) * V, cudaMemcpyDeviceToHost);
    free(pr_cpu);
}

void PersonalizedPageRank::cpu_validation(int iter) {

    // Reset the CPU PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr_golden.begin(), pr_golden.end(), 1.0 / V);

    // Do Personalized PageRank on CPU;
    auto start_tmp = clock_type::now();
    personalized_pagerank_cpu(x.data(), y.data(), val.data(), V, E, pr_golden.data(), dangling.data(), personalization_vertex, alpha, 1e-6, 100);
    auto end_tmp = clock_type::now();
    auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    std::cout << "exec time CPU=" << double(exec_time) / 1000 << " ms" << std::endl;

    // Obtain the vertices with highest PPR value;
    std::vector<std::pair<int, double>> sorted_pr_tuples = sort_pr(pr.data(), V);
    std::vector<std::pair<int, double>> sorted_pr_golden_tuples = sort_pr(pr_golden.data(), V);

    // Check how many of the correct top-20 PPR vertices are retrieved by the GPU;
    std::unordered_set<int> top_pr_indices;
    std::unordered_set<int> top_pr_golden_indices;
    int old_precision = std::cout.precision();
    std::cout.precision(4);
    int topk = std::min(V, topk_vertices);
    for (int i = 0; i < topk; i++) {
        int pr_id_gpu = sorted_pr_tuples[i].first;
        int pr_id_cpu = sorted_pr_golden_tuples[i].first;
        top_pr_indices.insert(pr_id_gpu);
        top_pr_golden_indices.insert(pr_id_cpu);
        if (debug) {
            double pr_val_gpu = sorted_pr_tuples[i].second;
            double pr_val_cpu = sorted_pr_golden_tuples[i].second;
            if (pr_id_gpu != pr_id_cpu) {
                std::cout << "* error in rank! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            } else if (std::abs(sorted_pr_tuples[i].second - sorted_pr_golden_tuples[i].second) > 1e-6) {
                std::cout << "* error in value! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            }
        }
    }
    std::cout.precision(old_precision);
    // Set intersection to find correctly retrieved vertices;
    std::vector<int> correctly_retrieved_vertices;
    set_intersection(top_pr_indices.begin(), top_pr_indices.end(), top_pr_golden_indices.begin(), top_pr_golden_indices.end(), std::back_inserter(correctly_retrieved_vertices));
    precision = double(correctly_retrieved_vertices.size()) / topk;
    if (debug) std::cout << "correctly retrived top-" << topk << " vertices=" << correctly_retrieved_vertices.size() << " (" << 100 * precision << "%)" << std::endl;
}

std::string PersonalizedPageRank::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(precision);
    } else {
        // Print the first few PageRank values (not sorted);
        std::ostringstream out;
        out.precision(3);
        out << "[";
        for (int i = 0; i < std::min(20, V); i++) {
            out << pr[i] << ", ";
        }
        out << "...]";
        return out.str();
    }
}

void PersonalizedPageRank::clean() {
    // Delete any GPU data or additional CPU data;
    // TODO!
}
