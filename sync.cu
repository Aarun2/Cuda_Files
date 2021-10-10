#pragma omp parallel shared(A, B, C)
{
    DoSomeWork(A, B);

// explicit barrier synchronization
#pragma omp barrier // wait till all threads finished

    DoSomeWork(B, C);

}

// parallel, necessary barrier can't be removed
// for, a for loop sits until all of them join
// single, only one thread executes within parallel region dont know which one
// master, I want only thread but it will id zero, no implicit synch point
// sections, like for, threads finish their section

// nowait clause proceed if done with your job
// applicable for for, single and sections

#pragma omp for schedule(static) nowait
    for (..)
        ...
// if done with for loop thread should not wait for others
#pragma omp for schedule(dynamic, 1)
    for (..)
        ...
        
// problem shared variable is being updated
// race condition no correct updates
#pragma omp parallel for shared(sum)
    for (...) { // i++ is happens at once
        sum +=
    }

// critical protects access to shared or modifiable data

#pragma omp critical(name)
    sum += ...;

#pragma omp parallel num_threads(4)

// atomic guarantees that the memory operation is going to happen atomically
// less overhead than critical

x[index[i]] // could have race condition as index[i] could return the same value

#pragma omp for shared(x)
    for (i ...)
        x[i] // this will have no race condition as each gets their on i value

#pragma omp atomic
    x[index[i]] // critical will also work but huge overhead

// atomic can only protect a single assignment, simple update of memory
// so x<op> = <expression>, op is atomic not the expression (also this must not reference x)
// increments and decrements are atomic
// op is +, -, * (ex:- +=)

// barriers are expensive
// there are explicit and implicit barriers
// be careful of nowait
// use critical or atomic and parallelize at the outermost level

// best is to use reduction clause

#pragma omp parallel for reduction (+:sum)
    for ()
        sum +=
// so each thread will have its own local copy of sum and finally added to globla variable
// + as sum + in for loop
// initially value is a 0 and 1 for multiply and so on
// +, *, -, &, |, ^, &&, || supported

// simd directive will ask one core/thread to do multiple operations
// 8 doubles and work on them
// single instruction and mutliple data

#pragma omp for simd reduction(+:sum)

// section will run a thread on each section
// only those number of sections will run even if more threads 
// so if 3 sections then one of the 4 threads will pick each up

// try to make less sequential
// load imbalance: one thread gets too much work

// use static schedule, most often with for loops, use schedule(runtime)

// setenv OMP_SCHEDULE "dynamic, 5" // setting the schedule

// try to parallelize the outer loop as inner loop will close and creates new context
// may create a load imbalance

// collapse for loops and parallelize that, better load balance
#pragma omp parallel for collapse(2)
for () {
    for () {
    
    }
}

// Symetric Multiprocessing (SMP) all threads/cores have similar overheads for accesing memory
// one chip has four cores so works well
// many chips or multi socket
// not all memory accesses are the same, not symetric
// non uniform access (NUMA)

// NUMA factor = largest/shortest average amount of time for a thread on a core to reach data in memory

// QPI (Quick path interconnect) (used inntel)








































