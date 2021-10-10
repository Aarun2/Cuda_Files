// can't use for with lists
// every thread is traversing the list below
// single thread process and the others move forward

#pragma omp parallel private(p)
{
    p = listhead;
    while (p != listend) {
        #pragma omp single nowait
        process(p);
        p = next(p);
    }

}

// single has a high cost, each thread needs to check back 
// to see if processed

// task generator will create tasks and threads will run them
// pragma to specify tasks and ensure there are no dependancies
// omp generates task

// code to execute, data environment, Internal Control Variables (ICV)
// each task has its own stack space that is destroyed when task completed
#pragma omp parallel
{
    #pragma omp single nowait
    {
        node *p = head_of_list;
        while (p != end_of_list) {
            // one thread posts all the tasks
            #pragma omp task firstprivate(p)
            process(p); // package this line
            p = p->next;
        }
    }
}

// if task B needs to wait for task A
// use thread or task barriers
// #pragma omp barrier
// #pragma omp taskwait

#pragma omp parallel
{
    #pragma omp task
    foo(); // many threads add task and complete it
    #pragma omp barrier // all foo tasks will be complted here
    #pragma omp single // only single thread adds task
    {
        #pragma omp task;
        bar();
    }
}

// sections are static figure out at compile time
// task dynamic figured out at run time

// shared and private variables
// any variable declared prior to parallel region is sahred
// other shared: global, file scope, namespace, const with no mutable, static

for private(x, y)

// lost outside the parallel block

// private variables are unintiallized
// loop iteration variables are private
// but threads don't run the same iteration values

firstprivate(i) // initialize with value outside block

lastprivate(i) // i at the end is given
// the value of the last thread

// default(none) require to specify scope of
// all outside of block
































