// cub fast library
// atomic: memory location updated atomically
// barrier: synchronizes threads, wait till all done
// critical: code is executed one thread at a time
// flush: all threads have the same view of memory for all shared objects
// for: loop is divided among threads
// master: only master thread should execute section
// ordered: code under parallel loop should be executed sequentially
// parallel: multiple threads in parallel
// sections: many sections picked up by different threads
// single: run by any one thread
// threadprivate: variable is private to thread

omp_[set|get]_num_threads() // set and get number of threads
omp_get_thread_num() // what is the thread num
omp_get_max_threads() // number of maximum threads

omp_in_parallel() // in paralllel?

omp_get_num_procs() // how many processors in system

omp_[set|unset]_lock() // explicit lock or unlock

#include<omp.h>

// set number of threads pertains to parallel regions 
// that follow not the sequential statement
// that will always be 1

omp_set_num_threads(2); // 2 set, runtime function call
print(omp_get_num_threads()); // prints 1

#pragma omp paralel // compiler directive
#pragma omp master
    { // only one print due to master
        printf(omp_get_num_threads()); // but prints 2
    } // as 2 threads declared

omp_get_max_threads(); // initially 12 or based on the hw
// but after set_num threads will reflect that

// environment variable
setenv OMP_NUM_THREADS 8 // C shell
export OMP_NUM_THREADS=8 // bash shell
set OMP_NUM_THREADS=8 // windows shell

// directives = suggestions, needs to be a structured block

// not good, not structured
if goto more
#pragma omp parallel {
    more:
} // implicit barrier here

// function wrapped around in a thunk
// generate needed pthreads
// call function
// join at the end

// nested parallelism
// thread spwans more threads
// that thread becomes the master of those threads

omp_set_nested(0) // dont allow nesting
// any other input allows nesting

omp_set_dynamic(1) // so can change nesting settings
omp_set_nested(1) // then change allow nesting to true

// set will pass to same and inner levels
// get will be for the same level only

// threads grows exponentially

#pragma omp parallel (3) {

    #pragma omp parallel (4) {
    
    }
}


for
    sum += f(i); // load imbalance if
    // for example as i increase f takes more time

schedule(static[, chunk])
// chunk number of iterations given to each thread
// Round Robin distribution
// low overhead but may cause load imbalance
// also false sharing if block size is too small
// so 2 threads try to edit the same cache line

schedule(dynamic[, chunk])
// grab chunk iterations, when done ask for more iterations
// high overhead but can reduce load imbalance
// problem cache misses as not in the same localcity
// if done quickly picks up data from different locality

schedule(guided[, chunk])
// dynamic schedule starting with large block
// size of the block shrinks, no smaller than chunk
// less expensive than dynamic but problem on loops
// where first iteration is the most expensive
    
parallel sections
{
    section
    ....
    section
    ...
}

// if num_threads < sections then one thread will be idel
// load imbalance

spin_in_place(2); // spin in place for 2 seconds

// tasks are for irregular problems
// like unbounded loops, recursive algos, prod/consumer


    
    
    
    
    
    
    
    
    
    




































