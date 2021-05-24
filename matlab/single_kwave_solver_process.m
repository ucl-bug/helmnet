function single_kwave_solver_process(k)
    load(['../tmp', num2str(k), '.mat'])

    [rows, cols] = size(sos_map);
    p = kwave_solver(sos_map, source_location, omega, min_sos, num2str(k));
    p = reshape(p, [rows, cols]);

    save(['../tmp', num2str(k), '.mat'])