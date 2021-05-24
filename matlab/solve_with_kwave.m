load("/tmp/helmholtz_setup.mat")

%addpath(genpath('/mnt/Software/k-Wave'))
addpath(genpath('~/repos/k-wave-matlab/k-Wave'))
[rows, cols] = size(sos_map);
[p, exec_time] = kwave_solver(sos_map, source_location, omega, min_sos, flag, cfl, roundtrips);
p = reshape(p, [rows, cols]);
disp('done')

save("/tmp/helmholtz.mat","p", "exec_time")