load("/tmp/helmholtz_setup.mat")

%addpath(genpath('/mnt/Software/k-Wave'))
addpath(genpath('~/repos/k-wave-matlab/k-Wave'))
addpath('matlab')
[rows, cols] = size(sos_map);
source_map = zeros(rows, cols);
sos_map = double(sos_map);
source_map(source_location(1)+1, source_location(2)+1) = 10.0;
[p, rel_error, A, B, M] = spectral_gmres_solver(sos_map,source_map,double(omega),double(pml_size),...
                            double(sigma_star), max_iter, restart, 0, 0);
disp('done')

save("/tmp/helmholtz.mat","p", "rel_error")