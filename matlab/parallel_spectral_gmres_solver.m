load("datasets/testset.mat");
speeds_of_sound = double(squeeze(speeds_of_sound(:,1,:,:)));

% Adding parameters
% TODO: this should be dynamic from the model
omega = 1.;
source_location = [82, 48];
pml_size = 8;
sigma_star = 2;
max_iter = 1000;
checkpoint_frequency = 100;

% Making source maps
source = zeros(size(speeds_of_sound,2)) + 0j;
source(source_location(1)+1, source_location(2)+1) = 10;

sos_map = squeeze(speeds_of_sound(1,:,:));

P = zeros(size(speeds_of_sound,1), 1+floor(max_iter/checkpoint_frequency), ...
    size(speeds_of_sound,2),size(speeds_of_sound,3)) + 0j;

disp("Running dummy solution to get matrices")
[p, relres, A, B] = spectral_gmres_solver(sos_map,source,omega,pml_size,sigma_star,5, ...
        5,0,0);
A = A;
B = B;

%parfor_progress(size(speeds_of_sound,1));
residuals = zeros(size(speeds_of_sound,1), 1+floor(max_iter/checkpoint_frequency));
parfor samplenum = 1:size(speeds_of_sound,1)
    sos_map = squeeze(speeds_of_sound(samplenum,:,:));
    [p, relres, temp, temp2] = spectral_gmres_solver(sos_map,source,omega,pml_size,sigma_star,max_iter, ...
        checkpoint_frequency,A,B);
    P(samplenum,:,:,:) = p; 
    residuals(samplenum,:) = relres;
    %parfor_progress;
end
%parfor_progress(0);

save 'gmres_results.mat'
    