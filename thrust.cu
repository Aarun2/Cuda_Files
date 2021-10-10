// Thrust = template library for GPU and CPU
// a header library

// C++ host side code

// allocate host memory
thrust::host_vector<int> h_vec(10);

// stl sort
std::sort(h_vec.begin(), h_vec.end());

// thrust sort
thrust::sort(h_vec.begin(), h_vec.end());

using namespace thrust;

int sum = reduce(h_vec.begin(), h_vec.end());

// host_vector stored in host memory
// device_vector stored in device memory
// mem copies take place for device vector
// but i don't have to free since i didnt allocate
// h_vec = d_vec // device to host copy

// no list in thrust but copy to copy to device vector available

thrust::device_vector<int> d_vec(h_list.size());

thrust::copy(h_list.begin(), h_list.end(), d_vec.begin());

thrust::device_vector<int> d_vec2((h_list.begin(), h_list.end());
// copy on constructor

// iterators can be incremented and can be converted to raw pointers

// iterator to pointer
int *ptr = thrust::raw_pointer_cast(&d_vec[0]);

// use ptr in CUDA C kernel
my_kernel<<< (N+255) / 256, 256 >>>(N, ptr);

cudaMemcpy(ptr, ...);

// pointer to iterator

// raw pointer to device memory
// so raw pointer points to allocated space
int *raw_ptr;
cudaMalloc((void **) &raw_ptr, N * sizeof(int));

// wrap raw pointer with a device pointer
// get equivalent thrust device pointer
thrust::device_ptr<int> dev_ptr(raw_ptr);

// initialize using device pointer
thrust::fill(dev_ptr, dev_ptr + N, (int) 0)

// access memory using device pointer
dev_ptr[0] = 1;

// free memory
cudaFree(raw_ptr); // since I allocated it

// Algorithms
// for_each, transform, gather, scatter
// reduce, inner_product, reduce_by_key
// inclusive_scan, inclusive_scan_by_key
// sort, stable_sort, sort_by_key


thrust::host_vector<int> h_vec(1 << 24); // 2^24

generate(h_vec.begin(), h_vec.end(), rand); // populate with 2^24 random number

device_vector<int> d_vec = h_vec; // or use copy

sort(d_vec.begin(), d_vec.end()); // sort on device and copy back

// copy back

// vector addition

device_vector<float> x(3);
device_vector<float> y(3);
device_vector<float> z(3);

x[0] = 10; x[1] = 20; x[2] = 30;
y[0] = 15; y[1] = 35; y[2] = 10;

// add x and y and store in z
// make sure dimensions match
transform(x.begin(), x.end(), y.begin(), z.begin(), thrust::plus<float>());

float init = x[0];

reduce(x.begin(), x.end(), init, maximum<float>()); // compare for maximum

// other functions are minus and multiply

// SAXPY = scalar add x plus y

// z = a *(x+y) todo

// functor, called for every pair
struct saxpy
{
    float m_a;
    
    saxpy(float a) : m_a(a) {}
    
    __host__ __device__
    float operator() (float x, float y)
    {
        return m_a * x + y;
    }
}

float aVal = 2.0f;

transform(x.begin(), x.end(), y.begin(), z.begin(), saxpy(aVal));

struct negate_float2
{
    __host__ __device__
    float operator() (float2 a)
    {
        return make_float2(-a.x, -a.y);
    }
};

device_vector<float2> input  = ...;
device_vector<float2> output  = ...;

negate_float2 fnctr;

// negate vectors
transform(input.begin(), input.end(), output.begin());

struct compare_float2
{
    __host__ __device__
    float operator() (float2 a, float2 b)
    {
        return a.x < b.x;
    }
};

device_vector<float2> vec = ...

compare_float2 comp;

sort(vec.begin(), vec.end(), comp);

// return true if x greater than threshold

struct is_greater_than
{
    int threshold;
    
    is_greater_than(int t) { threshold = t; }
    
    __host__ __device__
    bool operator()(int x) {return x > threshold; }
    
};

device_vector <int> vec = ...

is_greater_than pred(10);

// counts how many value greater than threshold and returns that
int result = count_if(vec.begin(), vec.end(), pred);

// other algorithms: reduce, find, mismatch, inner_product, equal, min_element
// count, is_sorted, transform_reduce

// what if more input transformations
// zipping
// combine input into pairs

// U = 2x + 2y + 2z
transform(make_zip_iterator(make_tuple(X.begin(), Y.begin(), Z.begin())),
          make_zip_iterator(make_tuple(X.end(), Y.end(), Z.end())),
          U.begin(),
          linear_combo());

struct linear_combo {
    __host__ __device__
    float operator() (tuple<float, float, float> t) {
        float x, y, z;
        tie(x, y, z) = t;
        return 2.0f*x + 2.0f*y + 2.0fz;
    }
};

// transform_reduce will do the equation and add up the U values like reduce
// add to whatever init value
// fusing operations = transform + reduce 
float myResult = transform_reduce(make_zip_iterator(make_tuple(X.begin(), Y.begin(), Z.begin())),
          make_zip_iterator(make_tuple(X.end(), Y.end(), Z.end())),
          linear_combo, init, plus<float>());

// peak flop rate: operations per second
// Max bandwidth: how much data can I bring in at the peak flop rate

// how many operations per byte of data = flop_rate/bw

// vector add, SAXPY ... are not at peak performance ~ 7:1
// vector_add = 1 : 12 =  1add and 4+4+4 data transfer

// SAXPY < FFT < SGEMM

// loop fusion compute U and V computation

// zipping = reorganize data
// fusing = reorganize computation for effeciency

struct linear_combo {
    __host__ __device__
    float operator() (tuple<float, float, float> t) {
        float x, y, z;
        tie(x, y, z) = t;
        float u = 2.0f*x + 2.0f*y + 2.0fz;
        float v = 3.0f*x + 4.0f*y + 5.0fz;
        return Tuple2(u, v);
    }
};

// u and v calculation fusion
transform(make_zip_iterator(make_tuple(X.begin(), Y.begin(), Z.begin())),
          make_zip_iterator(make_tuple(X.end(), Y.end(), Z.end())),
           make_zip_iterator(make_tuple(U.begin(), V.begin)),
          linear_combo());

// before bring 12 bytes return 4 bytes and bring 12 bytes return 4 bytes
// now bring 12 bytes return 8 bytes
// so 20 bytes transferred now for 32 operations
// so 32/20 is performance speedup 1.6

// square and sum it, lambda notation
int result = transform_reduce(x.begin(), x.end(), _1 *_1, 0.0f, plus<float>());





