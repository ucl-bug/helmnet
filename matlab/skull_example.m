if ~ismac
    addpath('/mnt/Software/k-Wave');
end

% downsampling factor for the geometry (starts at 512 x 512, so ds = 2 will
% give a domain size of 256 x 256)
ds = 1;
run_kw = true;

filename = '../examples/qure_ai-CQ500CT2-Unknown Study-CT0.625mm-CT000112.dcm';
inf = dicominfo(filename);
dat = dicomread(filename);
dx_true = inf.PixelSpacing(1) * 1e-3;

% downsample
dat = dat(1:ds:end, 1:ds:end);
dx_true = dx_true * ds;
[Nx, Ny] = size(dat);

% convert the skull map
settings.bg_sound_speed = 1500;
settings.tissue_sound_speed = 1500;
medium = skull2medium(single(dat) - 1000, [], 500e3, ...
    'ConversionSettings', settings, ...
    'SkullThreshold', 750);

% scale sound speed between 1 and 2
c = medium.sound_speed; 
c = c - min(c(:));
c = (c ./ max(c(:)) + 1);
label = '';
clear medium;

c0 = 1;
f = 1 / (2 * pi);
lambda = 2 * pi;
dx = 1;
sc = dx_true / dx;

% create grid
kgrid = kWaveGrid(Nx, dx, Ny, dx);

% set cfl and roundtrips
cfl = 0.01;
grid_traversals = 100;

% calculate the time step using an integer number of points per period
ppw = lambda / dx;                      % points per wavelength
ppp = ceil(ppw / cfl);                  % points per period
T   = 1 / f;                            % period [s]
dt  = T / ppp;                          % time step [s]

% calculate true frequency
c_true = 1500;
f_true = c_true / (ppw * dx_true);
disp(['True frequency = ' scaleSI(f_true) 'Hz']);

% calculate the number of time steps to reach steady state
t_end = grid_traversals * sqrt( kgrid.x_size.^2 + kgrid.y_size.^2 ) / c0; 
Nt = round(t_end / dt);

% create the time array
kgrid.setTime(Nt, dt);

% define the input signal
source.p = 10 * createCWSignals(kgrid.t_array, f, 1, 0);

% set the sensor mask to cover the entire grid
sensor.mask = ones(Nx, Ny);
sensor.record = {'p'};

% record the last 3 cycles in steady state
num_periods = 3;
sensor.record_start_index = Nt - num_periods * ppp + 1;

% setup transducer
roc_true = 60e-3;
diameter_true = 60e-3;

source.p_mask = makeArc([Nx, Ny], [430/ds, 380/ds], round(roc_true / (dx * sc)), round(diameter_true / (dx * sc)), [320/ds, 256/ds]);  

% define medium
medium.sound_speed = c;
medium.sound_speed_ref = c0;
medium.density = 1000 / 1500;

% set PML size
pml_size = 20;


% run simulation
sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, ...
    'PMLSize', pml_size);

% extract the pressure amplitude at each position
[p_kw_amp, p_kw_ph] = extractAmpPhase(sensor_data.p, 1/kgrid.dt, f);

% reshape the data
p_kw = p_kw_amp .* exp(1i * p_kw_ph);
p_kw = reshape(p_kw, Nx, Ny);

% save the data
save(['../examples/kwavedata' num2str(Nx) label], 'p_kw');

% save problem for pytorch
sos = medium.sound_speed;
src = source.p_mask;
save("../examples/problem_setup.mat", "sos", "src")