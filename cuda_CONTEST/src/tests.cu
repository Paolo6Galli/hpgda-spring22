



int main() {
    const int *x = {0,5,6,6,6,6,8,9,13};
    const int *y = {1,1,1,0,4,6,1,4,6};
    const double *val = {0,1.2,1,90.2,0.3,0.1,4.2,2.1,7.8};
    const double *vec ={0.1,4.2,2.1,7.8,0,1.2,1,90.2,0.3}
    double *resultcpu;
    float *resultgpu;
    int N = 9;
    personalized_pagerank.spmv_coo_cpu(x, y, val, vec, resultcpu, N);
    personalized_pagerank.spmv(x, y, val, vec, resultgpu, N);
    double cpu_result = resultcpu;
    float gpu_result = resultgpu;

    double max_err = 1e-6;

    for (int i = 0; i < N; i++) {
        double err = (double) gpu_result[i] - cpu_result[i];
        if err > max_err
            std::cout << "* error in value! (" << gpu_result[i] << ") correct=" << cpu_result[i] << std::endl;
    }
}