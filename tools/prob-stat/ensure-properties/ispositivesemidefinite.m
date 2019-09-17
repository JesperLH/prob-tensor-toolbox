function ispositivesemidefinite(A)
evals = eig(A);
assert(gather(all(evals>=0)))
evals=evals(evals<0);

assert(gather(all(abs(evals) <1e-12)))

end